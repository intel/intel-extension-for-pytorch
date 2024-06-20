#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ceil_div.h>
#include <ATen/native/Pool.h>
#include <vector>

#include <oneDNN/oneDNN.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
#include "utils/ComputeEngine.h"

using namespace dnnl;
using namespace at::native;
using namespace torch_ipex::xpu::dpcpp;
using namespace torch_ipex::xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t, typename accscalar_t>
struct AvgPool2dChannelsLastFrameKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();

    if (index < total_elements) {
      const int64_t c = index % channels;
      const int64_t pw = (index / channels) % pooled_width;
      const int64_t ph = (index / channels / pooled_width) % pooled_height;
      const int64_t n = index / channels / pooled_width / pooled_height;
      int64_t hstart = ph * stride_h - pad_h;
      int64_t wstart = pw * stride_w - pad_w;
      int64_t hend = Numerics<int64_t>::min(hstart + kernel_h, height + pad_h);
      int64_t wend = Numerics<int64_t>::min(wstart + kernel_w, width + pad_w);
      const int64_t pool_size = (hend - hstart) * (wend - wstart);
      hstart = Numerics<int64_t>::max(hstart, 0);
      wstart = Numerics<int64_t>::max(wstart, 0);
      hend = Numerics<int64_t>::min(hend, height);
      wend = Numerics<int64_t>::min(wend, width);

      if (hstart >= hend || wstart >= wend) {
        top_data[index] = scalar_t(0);
        return;
      }

      accscalar_t aveval = accscalar_t(0);
      const scalar_t* const bottom_slice =
          bottom_data + n * channels * height * width + c;
      for (int64_t h = hstart; h < hend; ++h) {
        for (int64_t w = wstart; w < wend; ++w) {
          aveval += bottom_slice[(h * width + w) * channels];
        }
      }
      int64_t divide_factor;
      if (use_divisor) {
        divide_factor = divisor_override;
      } else {
        if (count_include_pad) {
          divide_factor = pool_size;
        } else {
          divide_factor = (hend - hstart) * (wend - wstart);
        }
      }
      top_data[index] = static_cast<scalar_t>(aveval / divide_factor);
    }
  }
  AvgPool2dChannelsLastFrameKernelFunctor(
      scalar_t* top_data_,
      const scalar_t* bottom_data_,
      int64_t total_elements_,
      int64_t group_size_,
      int64_t num_groups_,
      int64_t channels_,
      int64_t height_,
      int64_t width_,
      int64_t pooled_height_,
      int64_t pooled_width_,
      int64_t kernel_h_,
      int64_t kernel_w_,
      int64_t stride_h_,
      int64_t stride_w_,
      int64_t pad_h_,
      int64_t pad_w_,
      int64_t divisor_override_,
      bool count_include_pad_,
      bool use_divisor_)
      : top_data(top_data_),
        bottom_data(bottom_data_),
        total_elements(total_elements_),
        group_size(group_size_),
        num_groups(num_groups_),
        channels(channels_),
        height(height_),
        width(width_),
        pooled_height(pooled_height_),
        pooled_width(pooled_width_),
        kernel_h(kernel_h_),
        kernel_w(kernel_w_),
        stride_h(stride_h_),
        stride_w(stride_w_),
        pad_h(pad_h_),
        pad_w(pad_w_),
        divisor_override(divisor_override_),
        count_include_pad(count_include_pad_),
        use_divisor(use_divisor_) {}

 private:
  scalar_t* top_data;
  const scalar_t* bottom_data;
  int64_t total_elements;
  int64_t group_size;
  int64_t num_groups;
  int64_t channels;
  int64_t height;
  int64_t width;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t kernel_h;
  int64_t kernel_w;
  int64_t stride_h;
  int64_t stride_w;
  int64_t pad_h;
  int64_t pad_w;
  int64_t divisor_override;
  bool count_include_pad;
  bool use_divisor;
};

template <typename scalar_t, typename accscalar_t>
void avg_pool2d_channels_last_frame(
    const Tensor& bottom_data_,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    Tensor& top_data_,
    const int64_t divisor_override,
    const bool count_include_pad,
    const bool use_divisor) {
  scalar_t* top_data = top_data_.data_ptr<scalar_t>();
  const scalar_t* bottom_data = bottom_data_.data_ptr<scalar_t>();
  const int64_t total_elements = top_data_.numel();
  const int64_t group_size = dpcppMaxWorkGroupSize();
  const int64_t num_groups = ceil_div<int64_t>(total_elements, group_size);

  auto cgf = DPCPP_Q_CGF(cgh) {
    AvgPool2dChannelsLastFrameKernelFunctor<scalar_t, accscalar_t> kfn(
        top_data,
        bottom_data,
        total_elements,
        group_size,
        num_groups,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        divisor_override,
        count_include_pad,
        use_divisor);
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
  };

  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
}

template <typename scalar_t, typename accscalar_t>
struct AvgPool2dOutFrameKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();

    if (index < total_elements) {
      const int64_t pw = index % pooled_width;
      const int64_t ph = (index / pooled_width) % pooled_height;
      const int64_t c = (index / pooled_width / pooled_height) % channels;
      const int64_t n = index / pooled_width / pooled_height / channels;

      int64_t hstart = ph * stride_h - pad_h;
      int64_t wstart = pw * stride_w - pad_w;
      int64_t hend = Numerics<int64_t>::min(hstart + kernel_h, height + pad_h);
      int64_t wend = Numerics<int64_t>::min(wstart + kernel_w, width + pad_w);
      const int64_t pool_size = (hend - hstart) * (wend - wstart);
      hstart = Numerics<int64_t>::max(hstart, 0);
      wstart = Numerics<int64_t>::max(wstart, 0);
      hend = Numerics<int64_t>::min(hend, height);
      wend = Numerics<int64_t>::min(wend, width);

      if (hstart >= hend || wstart >= wend) {
        top_data[index] = scalar_t(0);
        return;
      }

      accscalar_t aveval = accscalar_t(0);
      const scalar_t* const bottom_slice =
          bottom_data + (n * channels + c) * height * width;

      for (int64_t h = hstart; h < hend; ++h) {
        for (int64_t w = wstart; w < wend; ++w) {
          aveval += bottom_slice[h * width + w];
        }
      }
      int64_t divide_factor;
      if (use_divisor) {
        divide_factor = divisor_override;
      } else {
        if (count_include_pad) {
          divide_factor = pool_size;
        } else {
          divide_factor = (hend - hstart) * (wend - wstart);
        }
      }
      top_data[index] = static_cast<scalar_t>(aveval / divide_factor);
    }
  }
  AvgPool2dOutFrameKernelFunctor(
      scalar_t* top_data_,
      const scalar_t* bottom_data_,
      int64_t total_elements_,
      int64_t group_size_,
      int64_t num_groups_,
      int64_t channels_,
      int64_t height_,
      int64_t width_,
      int64_t pooled_height_,
      int64_t pooled_width_,
      int64_t kernel_h_,
      int64_t kernel_w_,
      int64_t stride_h_,
      int64_t stride_w_,
      int64_t pad_h_,
      int64_t pad_w_,
      int64_t divisor_override_,
      bool count_include_pad_,
      bool use_divisor_)
      : top_data(top_data_),
        bottom_data(bottom_data_),
        total_elements(total_elements_),
        group_size(group_size_),
        num_groups(num_groups_),
        channels(channels_),
        height(height_),
        width(width_),
        pooled_height(pooled_height_),
        pooled_width(pooled_width_),
        kernel_h(kernel_h_),
        kernel_w(kernel_w_),
        stride_h(stride_h_),
        stride_w(stride_w_),
        pad_h(pad_h_),
        pad_w(pad_w_),
        divisor_override(divisor_override_),
        count_include_pad(count_include_pad_),
        use_divisor(use_divisor_) {}

 private:
  scalar_t* top_data;
  const scalar_t* bottom_data;
  int64_t total_elements;
  int64_t group_size;
  int64_t num_groups;
  int64_t channels;
  int64_t height;
  int64_t width;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t kernel_h;
  int64_t kernel_w;
  int64_t stride_h;
  int64_t stride_w;
  int64_t pad_h;
  int64_t pad_w;
  int64_t divisor_override;
  bool count_include_pad;
  bool use_divisor;
};

template <typename scalar_t, typename accscalar_t>
void avg_pool2d_out_frame(
    const Tensor& bottom_data_,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    Tensor& top_data_,
    const int64_t divisor_override,
    const bool count_include_pad,
    const bool use_divisor) {
  scalar_t* top_data = top_data_.data_ptr<scalar_t>();
  const scalar_t* bottom_data = bottom_data_.data_ptr<scalar_t>();

  const int64_t total_elements = top_data_.numel();
  const int64_t group_size = dpcppMaxWorkGroupSize();
  const int64_t num_groups = ceil_div<int64_t>(total_elements, group_size);

  auto cgf = DPCPP_Q_CGF(cgh) {
    AvgPool2dOutFrameKernelFunctor<scalar_t, accscalar_t> kfn(
        top_data,
        bottom_data,
        total_elements,
        group_size,
        num_groups,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        divisor_override,
        count_include_pad,
        use_divisor);

    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
}

void avg_pool2d_out_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);
  std::vector<int64_t> kernel_vec = {kH, kW};

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);
  std::vector<int64_t> stride_vec = {dH, dW};

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of "
      "two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);
  std::vector<int64_t> padding_vec = {padH, padW};

  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);

  const auto outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
  const auto outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);

  const auto memory_format = input.suggest_memory_format();

  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value =
      use_divisor ? divisor_override.value() : 0;

  /* PyTorch support two cases of AvgPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the suggest_memory_format can only be Contiguous or ChannelsLast1D
     (nwc), the ChannelsLast1D (nwc) does not match the sementics of Input (C,
     H, W) case. Then the suggest_memory_format can only be Contiguous.
     2. 4D: Input (N, C, H, W),  Output (N, C, H0, W0), Kernel (kH, kW)
     This case supports Contiguous and ChannelsLast2D memory_format. */

  /* get contiguous input */
  Tensor input_ = input.ndimension() == 3
      ? input.contiguous()
      : contiguous_if_needed(input, input.suggest_memory_format());

  pool2d_shape_check(
      input_,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      1,
      1,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      memory_format);

  /* resize output/indices */
  if (input.ndimension() == 3) {
    output.resize_({nInputPlane, outputHeight, outputWidth}, memory_format);
  } else {
    output.resize_(
        {nbatch, nInputPlane, outputHeight, outputWidth}, memory_format);
  }

  // per oneDNN definition, no dilation means dilation ratio is 0
  std::vector<int64_t> dilation_vec = {0, 0};
  torch_ipex::xpu::COMPUTE_ENG real_eng =
      choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::BASIC, input);

  // for onednn block format
  if (torch_ipex::xpu::COMPUTE_ENG::ONEDNN == real_eng) {
    if (count_include_pad) {
      torch_ipex::xpu::oneDNN::pooling<alg::pooling_avg_include_padding>(
          output,
          input_,
          nbatch,
          nInputPlane,
          0,
          inputHeight,
          inputWidth,
          0,
          outputHeight,
          outputWidth,
          stride_vec,
          kernel_vec,
          dilation_vec,
          padding_vec,
          padding_vec);
    } else {
      torch_ipex::xpu::oneDNN::pooling<alg::pooling_avg_exclude_padding>(
          output,
          input_,
          nbatch,
          nInputPlane,
          0,
          inputHeight,
          inputWidth,
          0,
          outputHeight,
          outputWidth,
          stride_vec,
          kernel_vec,
          dilation_vec,
          padding_vec,
          padding_vec);
    }
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        input.scalar_type(),
        "avg_pool2d_out_frame",
        [&] {
          using accscalar_t = acc_type<scalar_t>;

          if (at::MemoryFormat::ChannelsLast == memory_format) {
            avg_pool2d_channels_last_frame<scalar_t, accscalar_t>(
                input_,
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                kH,
                kW,
                dH,
                dW,
                padH,
                padW,
                output,
                divisor_override_value,
                count_include_pad,
                use_divisor);
          } else {
            // use contiguous memory format as default path
            input_ = input_.contiguous();
            avg_pool2d_out_frame<scalar_t, accscalar_t>(
                input_,
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                kH,
                kW,
                dH,
                dW,
                padH,
                padW,
                output,
                divisor_override_value,
                count_include_pad,
                use_divisor);
          }
        });
  }
}
template <typename scalar_t, typename accscalar_t>
struct AvgPool2dChannelsLastBackwardOutKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t index = item.get_global_linear_id();
    if (index < total_elements) {
      const int c = index % channels;
      const int w = (index / channels) % width + pad_w;
      const int h = (index / channels / width) % height + pad_h;
      const int n = index / channels / width / height;
      const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
      const int phend = Numerics<int>::min(h / stride_h + 1, pooled_height);
      const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
      const int pwend = Numerics<int>::min(w / stride_w + 1, pooled_width);
      accscalar_t gradient = accscalar_t(0);
      const scalar_t* const top_slice =
          top_data + n * channels * pooled_height * pooled_width + c;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          // figure out the pooling size
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = Numerics<int>::min(hstart + kernel_h, height + pad_h);
          int wend = Numerics<int>::min(wstart + kernel_w, width + pad_w);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = Numerics<int>::max(hstart, 0);
          wstart = Numerics<int>::max(wstart, 0);
          hend = Numerics<int>::min(hend, height);
          wend = Numerics<int>::min(wend, width);
          if (hstart >= hend || wstart >= wend) {
            continue;
          }
          int divide_factor;
          if (use_divisor) {
            divide_factor = divisor_override;
          } else {
            if (count_include_pad) {
              divide_factor = pool_size;
            } else {
              divide_factor = (hend - hstart) * (wend - wstart);
            }
          }
          gradient +=
              top_slice[(ph * pooled_width + pw) * channels] / divide_factor;
        }
      }
      bottom_data[index] = static_cast<scalar_t>(gradient);
    }
  }
  AvgPool2dChannelsLastBackwardOutKernelFunctor(
      const scalar_t* top_data_,
      scalar_t* bottom_data_,
      int64_t total_elements_,
      int64_t channels_,
      int64_t height_,
      int64_t width_,
      int64_t pooled_height_,
      int64_t pooled_width_,
      int kernel_h_,
      int kernel_w_,
      int stride_h_,
      int stride_w_,
      int pad_h_,
      int pad_w_,
      int divisor_override_,
      bool count_include_pad_,
      bool use_divisor_)
      : top_data(top_data_),
        bottom_data(bottom_data_),
        total_elements(total_elements_),
        channels(channels_),
        height(height_),
        width(width_),
        pooled_height(pooled_height_),
        pooled_width(pooled_width_),
        kernel_h(kernel_h_),
        kernel_w(kernel_w_),
        stride_h(stride_h_),
        stride_w(stride_w_),
        pad_h(pad_h_),
        pad_w(pad_w_),
        divisor_override(divisor_override_),
        count_include_pad(count_include_pad_),
        use_divisor(use_divisor_) {}

 private:
  const scalar_t* top_data;
  scalar_t* bottom_data;
  int64_t total_elements;
  int64_t channels;
  int64_t height;
  int64_t width;
  int64_t pooled_height;
  int64_t pooled_width;
  int kernel_h;
  int kernel_w;
  int stride_h;
  int stride_w;
  int pad_h;
  int pad_w;
  int divisor_override;
  bool count_include_pad;
  bool use_divisor;
};
template <typename scalar_t, typename accscalar_t>
void avg_pool2d_backward_out_channels_last_frame(
    const Tensor& grad_output,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    Tensor& grad_input,
    const int64_t divisor_override,
    bool count_include_pad,
    bool use_divisor) {
  auto top_data = grad_output.data_ptr<scalar_t>();
  auto bottom_data = grad_input.data_ptr<scalar_t>();
  const int64_t total_elems_count = grad_input.numel();
  const int64_t group_size =
      dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
  const int64_t num_groups = ceil_div<int64_t>(total_elems_count, group_size);
  auto cgf = DPCPP_Q_CGF(cgh) {
    AvgPool2dChannelsLastBackwardOutKernelFunctor<scalar_t, accscalar_t> kfn(
        top_data,
        bottom_data,
        total_elems_count,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        divisor_override,
        count_include_pad,
        use_divisor);
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
}
template <typename scalar_t, typename accscalar_t>
struct AvgPool2dBackwardOutKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t index = item.get_global_linear_id();
    if (index < total_elements) {
      // find out the local index
      // find out the local offset
      const int w = index % width + pad_w;
      const int h = (index / width) % height + pad_h;
      const int c = (index / width / height) % channels;
      const int n = index / width / height / channels;
      const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
      const int phend = Numerics<int>::min(h / stride_h + 1, pooled_height);
      const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
      const int pwend = Numerics<int>::min(w / stride_w + 1, pooled_width);
      accscalar_t gradient = accscalar_t(0);
      const scalar_t* const top_data_slice =
          top_data + (n * channels + c) * pooled_height * pooled_width;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          // figure out the pooling size
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = Numerics<int>::min(hstart + kernel_h, height + pad_h);
          int wend = Numerics<int>::min(wstart + kernel_w, width + pad_w);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = Numerics<int>::max(hstart, 0);
          wstart = Numerics<int>::max(wstart, 0);
          hend = Numerics<int>::min(hend, height);
          wend = Numerics<int>::min(wend, width);
          if (hstart >= hend || wstart >= wend) {
            continue;
          }
          int divide_factor;
          if (use_divisor) {
            divide_factor = divisor_override;
          } else {
            if (count_include_pad) {
              divide_factor = pool_size;
            } else {
              divide_factor = (hend - hstart) * (wend - wstart);
            }
          }
          gradient += top_data_slice[ph * pooled_width + pw] / divide_factor;
        }
      }
      bottom_data[index] = static_cast<scalar_t>(gradient);
    }
  }
  AvgPool2dBackwardOutKernelFunctor(
      const scalar_t* top_data_,
      scalar_t* bottom_data_,
      int64_t total_elements_,
      int64_t channels_,
      int64_t height_,
      int64_t width_,
      int64_t pooled_height_,
      int64_t pooled_width_,
      int kernel_h_,
      int kernel_w_,
      int stride_h_,
      int stride_w_,
      int pad_h_,
      int pad_w_,
      int divisor_override_,
      bool count_include_pad_,
      bool use_divisor_)
      : top_data(top_data_),
        bottom_data(bottom_data_),
        total_elements(total_elements_),
        channels(channels_),
        height(height_),
        width(width_),
        pooled_height(pooled_height_),
        pooled_width(pooled_width_),
        kernel_h(kernel_h_),
        kernel_w(kernel_w_),
        stride_h(stride_h_),
        stride_w(stride_w_),
        pad_h(pad_h_),
        pad_w(pad_w_),
        divisor_override(divisor_override_),
        count_include_pad(count_include_pad_),
        use_divisor(use_divisor_) {}

 private:
  const scalar_t* top_data;
  scalar_t* bottom_data;
  int64_t total_elements;
  int64_t channels;
  int64_t height;
  int64_t width;
  int64_t pooled_height;
  int64_t pooled_width;
  int kernel_h;
  int kernel_w;
  int stride_h;
  int stride_w;
  int pad_h;
  int pad_w;
  int divisor_override;
  bool count_include_pad;
  bool use_divisor;
};
template <typename scalar_t, typename accscalar_t>
void avg_pool2d_backward_out_frame(
    const Tensor& grad_output,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    Tensor& grad_input,
    const int64_t divisor_override,
    bool count_include_pad,
    bool use_divisor) {
  auto top_data = grad_output.data_ptr<scalar_t>();
  auto bottom_data = grad_input.data_ptr<scalar_t>();
  const int64_t total_elems_count = grad_input.numel();
  const int64_t group_size =
      dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue());
  const int64_t num_groups = ceil_div<int64_t>(total_elems_count, group_size);
  auto cgf = DPCPP_Q_CGF(cgh) {
    AvgPool2dBackwardOutKernelFunctor<scalar_t, accscalar_t> kfn(
        top_data,
        bottom_data,
        total_elems_count,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        divisor_override,
        count_include_pad,
        use_divisor);
    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
  };
  DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
}
Tensor& avg_pool2d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    torch_ipex::xpu::COMPUTE_ENG real_eng) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);
  std::vector<int64_t> kernel_vec = {kH, kW};

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);
  std::vector<int64_t> stride_vec = {dH, dW};

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of "
      "two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);
  std::vector<int64_t> padding_vec = {padH, padW};

  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value =
      use_divisor ? divisor_override.value() : 0;

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);
  const auto outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  const auto outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  auto memory_format = input.suggest_memory_format();
  avg_pool2d_backward_shape_check(
      input,
      gradOutput,
      nbatch,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      memory_format);

  // per oneDNN definition, no dilation means dilation ratio is 0
  std::vector<int64_t> dilation_vec = {0, 0};

  // xpu::COMPUTE_ENG real_eng =
  //     choose_compute_eng(xpu::COMPUTE_ENG::BASIC, input);

  // for onednn block format
  if (torch_ipex::xpu::COMPUTE_ENG::ONEDNN == real_eng) {
    if (count_include_pad) {
      torch_ipex::xpu::oneDNN::pooling_backward<
          alg::pooling_avg_include_padding>(
          gradInput,
          gradOutput,
          input,
          nbatch,
          nInputPlane,
          0,
          inputHeight,
          inputWidth,
          0,
          outputHeight,
          outputWidth,
          stride_vec,
          kernel_vec,
          dilation_vec,
          padding_vec,
          padding_vec);
    } else {
      torch_ipex::xpu::oneDNN::pooling_backward<
          alg::pooling_avg_exclude_padding>(
          gradInput,
          gradOutput,
          input,
          nbatch,
          nInputPlane,
          0,
          inputHeight,
          inputWidth,
          0,
          outputHeight,
          outputWidth,
          stride_vec,
          kernel_vec,
          dilation_vec,
          padding_vec,
          padding_vec);
    }
  } else {
    const Tensor gradOutput_ = gradOutput.contiguous(memory_format);
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        input.scalar_type(),
        "avg_pool2d_out_frame",
        [&] {
          using accscalar_t = acc_type<scalar_t>;
          if (memory_format == MemoryFormat::ChannelsLast) {
            gradInput.unsafeGetTensorImpl()->empty_tensor_restride(
                MemoryFormat::ChannelsLast);
            avg_pool2d_backward_out_channels_last_frame<scalar_t, accscalar_t>(
                gradOutput_,
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                kH,
                kW,
                dH,
                dW,
                padH,
                padW,
                gradInput,
                divisor_override_value,
                count_include_pad,
                use_divisor);
          } else {
            avg_pool2d_backward_out_frame<scalar_t, accscalar_t>(
                gradOutput_,
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                kH,
                kW,
                dH,
                dW,
                padH,
                padW,
                gradInput,
                divisor_override_value,
                count_include_pad,
                use_divisor);
          }
        });
  }
  return gradInput;
}

} // namespace impl

Tensor& avg_pool2d_out(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& output) {
  impl::avg_pool2d_out_template(
      output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
  return output;
}

Tensor& avg_pool2d_backward_out(
    const Tensor& grad_output_,
    const Tensor& self_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& grad_input) {
  // TORCH_CHECK(
  //     !divisor_override.has_value(),
  //     "dpcpp_avg_pool2d operator does not support divisor");

  /* PyTorch support two cases of AvgPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the suggest_memory_format can only be Contiguous or ChannelsLast1D
     (nwc), the ChannelsLast1D (nwc) does not match the sementics of Input (C,
     H, W) case. Then the suggest_memory_format can only be Contiguous.
     2. 4D: Input (N, C, H, W),  Output (N, C, H0, W0), Kernel (kH, kW)
     This case supports Contiguous and ChannelsLast2D memory_format. */
  torch_ipex::xpu::COMPUTE_ENG real_eng =
      choose_compute_eng(torch_ipex::xpu::COMPUTE_ENG::BASIC, grad_output_);
  Tensor self, grad_output;
  if (self_.ndimension() == 3) {
    self = self_.contiguous();
    grad_output = grad_output_.contiguous();
    grad_input.resize_as_(self);
  } else {
    auto smf = self_.suggest_memory_format();
    self = contiguous_if_needed(self_, smf);
    grad_output = contiguous_if_needed(grad_output_, smf);
    grad_input.resize_as_(self, smf);
  }

  impl::avg_pool2d_backward_out_template(
      grad_input,
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override,
      real_eng);
  return grad_input;
}
} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
Tensor avg_pool2d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  Tensor output;
  output = at::_empty_affine_quantized(
      {0},
      input.options(),
      input.q_scale(),
      input.q_zero_point(),
      MemoryFormat::Contiguous);

  return at::AtenIpexTypeXPU::avg_pool2d_out(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override,
      output);
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
