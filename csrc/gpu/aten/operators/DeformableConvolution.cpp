#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <c10/core/ScalarType.h>
#include <oneDNN/oneDNN.h>
#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
#include "oneDNN/Attr.h"
#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {
using xpu::oneDNN::Attr;

Tensor _convolution(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    Attr attr);

template <typename scalar_t>
void deform_interpolate(
    scalar_t* input,
    scalar_t* out,
    scalar_t* offset,
    scalar_t* mask,
    const int batch,
    const int channel,
    const int im_h,
    const int im_w,
    const int k_h,
    const int k_w,
    const int o_h,
    const int o_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group) {
  int channel_per_group = channel / deformable_group;

  int64_t out_size = o_h * o_w * channel * batch;
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);

  // Initialize workitem for every convolution's output pixel.
  int64_t wg_size = std::min(max_wg_size, out_size);
  int64_t wg_num = (out_size + max_wg_size - 1) / wg_size;
  int64_t glb_size = wg_num * wg_size;
  auto cgf = DPCPP_Q_CGF(_cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
      auto local_id = item_id.get_local_id(0);
      auto local_range = item_id.get_local_range(0);
      auto wg_id = item_id.get_group(0);
      auto wg_range = item_id.get_group_range(0);
      auto idx = item_id.get_global_linear_id();
      // for (int64_t i = idx; i < out_size; i += glb_size) {
      if (idx < out_size) {
        // 1. calculate the index of input and output for current work item.
        int64_t tmp = idx;
        int o_w_idx = tmp % o_w;
        int im_w_idx = o_w_idx * stride_w - pad_w;
        tmp /= o_w;
        int o_h_idx = tmp % o_h;
        int im_h_idx = o_h_idx * stride_h - pad_h;
        tmp /= o_h;
        int im_cn_idx = tmp % channel;
        int deform_idx = im_cn_idx / channel_per_group;
        int im_bs_idx = tmp / channel;

        // 2. calculate the offset index for each pointer.
        int64_t offset_idx = (im_bs_idx * deformable_group + deform_idx) * k_w *
            k_h * o_h * o_w * 2;
        int64_t mask_idx =
            (im_bs_idx * deformable_group + deform_idx) * k_w * k_h * o_h * o_w;
        int64_t im_idx = (im_bs_idx * channel + im_cn_idx) * im_h * im_w;
        scalar_t* img_ptr = input + im_idx;
        scalar_t* offset_ptr = offset + offset_idx;
        scalar_t* mask_ptr = mask + mask_idx;
        int64_t deform_im_idx =
            (im_bs_idx * channel + im_cn_idx) * o_h * k_h * o_w * k_w +
            o_h_idx * o_w * k_w * k_h + o_w_idx * k_w;

        // 3. Data chunk with size of kernel_h * kernel_w is calculated by each
        // work item, each data chunk will bond to the specific dcn output
        // pixel.
        for (int j = 0; j < k_h; ++j) {
          for (int k = 0; k < k_w; ++k) {
            int local_offset_h_idx = 2 * (j * k_w + k);
            int local_offset_w_idx = local_offset_h_idx + 1;
            local_offset_h_idx =
                (local_offset_h_idx * o_h + o_h_idx) * o_w + o_w_idx;
            local_offset_w_idx =
                (local_offset_w_idx * o_h + o_h_idx) * o_w + o_w_idx;
            int local_mask_idx =
                ((j * k_w + k) * o_h + o_h_idx) * o_w + o_w_idx;
            scalar_t local_offset_w = offset_ptr[local_offset_w_idx];
            scalar_t local_offset_h = offset_ptr[local_offset_h_idx];

            scalar_t mask = mask_ptr[local_mask_idx];
            scalar_t h = im_h_idx + j * dilation_h + local_offset_h;
            scalar_t w = im_w_idx + k * dilation_w + local_offset_w;

            int local_offset = j * k_w * o_w + k;
            int cur_df_idx = deform_im_idx + local_offset;

            scalar_t res = 0;

            // 4. For every pixel that located in the input feature map,
            // bilinear interpolation is performed according to the offset_ptr's
            // data.
            if (h > -1 && w > -1 && h < im_h && w < im_w) {
              int h_l_idx = Numerics<scalar_t>::floor(h), h_h_idx = h_l_idx + 1;
              int w_l_idx = Numerics<scalar_t>::floor(w), w_h_idx = w_l_idx + 1;
              scalar_t h_lu = (h_l_idx >= 0 && w_l_idx >= 0)
                  ? img_ptr[h_l_idx * im_w + w_l_idx]
                  : static_cast<scalar_t>(0);
              scalar_t h_lb = (h_h_idx <= im_h - 1 && w_l_idx >= 0)
                  ? img_ptr[h_h_idx * im_w + w_l_idx]
                  : static_cast<scalar_t>(0);
              scalar_t h_ru = (h_l_idx >= 0 && w_h_idx <= im_w - 1)
                  ? img_ptr[h_l_idx * im_w + w_h_idx]
                  : static_cast<scalar_t>(0);
              scalar_t h_rb = (h_h_idx <= im_h - 1 && w_h_idx <= im_w - 1)
                  ? img_ptr[h_h_idx * im_w + w_h_idx]
                  : static_cast<scalar_t>(0);
              scalar_t h_h_w = h - h_l_idx, h_l_w = 1 - h_h_w;
              scalar_t w_h_w = w - w_l_idx, w_l_w = 1 - w_h_w;
              scalar_t w_lu = h_l_w * w_l_w, w_lb = h_h_w * w_l_w;
              scalar_t w_ru = h_l_w * w_h_w, w_rb = h_h_w * w_h_w;
              res = h_lu * w_lu + h_lb * w_lb + h_ru * w_ru + h_rb * w_rb;
            }
            out[cur_df_idx] = res * mask;
          }
        }
      }
    };
    _cgh.parallel_for(sycl::nd_range<1>({glb_size}, {wg_size}), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

Tensor deform_interpolation_forward(
    const Tensor& input,
    const Tensor& offset,
    const Tensor& mask,
    const int64_t k_h,
    const int64_t k_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    const int64_t deformable_group) {
  int bs = input.size(0), cn = input.size(1);
  int im_h = input.size(2), im_w = input.size(3);
  int o_h = (im_h + pad_h * 2 - (dilation_h * (k_h - 1) + 1)) / stride_h + 1;
  int o_w = (im_w + pad_w * 2 - (dilation_w * (k_w - 1) + 1)) / stride_w + 1;
  Tensor out = at::empty({bs, cn, o_h * k_h, o_w * k_w}, input.options());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "deform_image",
      [&]() {
        deform_interpolate<scalar_t>(
            input.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            offset.data_ptr<scalar_t>(),
            mask.data_ptr<scalar_t>(),
            bs,
            cn,
            im_h,
            im_w,
            k_h,
            k_w,
            o_h,
            o_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            deformable_group);
      });
  return out;
}

Tensor dcn_v2_forward(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const Tensor& offset,
    const Tensor& mask,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    const int64_t deformable_group) {
  int bs = input.size(0), cn = input.size(1);
  int im_h = input.size(2), im_w = input.size(3);
  int k_h = weight.size(2), k_w = weight.size(3);
  TORCH_INTERNAL_ASSERT(
      k_h == kernel_h,
      "deformable convolution: The kernel height do not match its tensor shape");
  TORCH_INTERNAL_ASSERT(
      k_w == kernel_w,
      "deformable convolution: The kernel width do not match its tensor shape");
  TORCH_INTERNAL_ASSERT(
      cn % deformable_group == 0,
      "deformable group should be dividable by input channel");

  // We decouple the deformable convolution to two parts:
  //  1. deform the image:
  //    according to the input parameter, adopt bilinear interpolation to create
  //    the deformed feature map for convolution. The deformed image should have
  //    size: [ batchsize, channel, output_h * kernel_h, output_w * kernel_w ]
  //  2. norm convolution:
  //    execute convlution with calculatde kernel size, pading size, dilation
  //    size and stride to get the final result.

  Tensor deform_img = deform_interpolation_forward(
      input,
      offset,
      mask,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      deformable_group);

  Attr att;
  Tensor _bias = bias.has_value() ? bias.value() : Tensor();
  Tensor ret = _convolution(
      deform_img,
      weight,
      _bias,
      {kernel_h, kernel_w},
      {0, 0},
      {1, 1},
      false,
      {0, 0},
      1,
      att);
  return ret;
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_NEED_PLAIN("dcn_v2_forward", dcn_v2_forward);
  IPEX_OP_REGISTER_NEED_PLAIN(
      "deform_interpolation_forward", deform_interpolation_forward);
}
} // namespace AtenIpexTypeXPU

} // namespace at