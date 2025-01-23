#include <ATen/native/UpSample.h>
#include <core/MemoryFormat.h>
#include <oneDNN/oneDNN.h>
#include <tensor/Tensor.h>
#include "UpSample.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"

using namespace dnnl;
using namespace at::native;
using namespace at::AtenIpexTypeXPU;
using namespace torch_ipex::xpu::dpcpp;
using namespace torch_ipex::xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {

using namespace impl;
using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

struct Nearest_index_op {
  int operator()(const float scale, int dst_index, int input_size) const {
    // index_f32 = (output_index) * scale
    // input_index = round(index_f32)
    // Same as a buggy OpenCV INTER_NEAREST
    // We keep this method for BC and consider as deprecated.
    // See nearest_exact as replacement
    const int src_index =
        min(static_cast<int>(floorf((dst_index)*scale)), input_size - 1);
    return src_index;
  }
};

// struct Nearest_bw_index_op {
//   int operator()(const float scale, int dst_index, int output_size) const {
//     // Equivalent to buggy OpenCV INTER_NEAREST
//     // We keep this method for BC and consider as deprecated.
//     // See nearest_exact as replacement
//     const int src_index =
//         min(static_cast<int>(ceilf(dst_index * scale)), output_size);
//     return src_index;
//   }
// };

static int lastPow2(unsigned int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max<int>(1, n - (n >> 1));
}

inline size_t idx_cl(
    const size_t n,
    const size_t h,
    const size_t w,
    const size_t c,
    const size_t height,
    const size_t width,
    const size_t channel) {
  return ((n * height + h) * width + w) * channel + c;
}

template <typename scalar_t, typename index_op_t>
struct UpsampleNearest2dOutKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    size_t nc_idx = item.get_global_id(0);
    int h2 = item.get_global_id(1);
    int w2 = item.get_global_id(2);

    if (w2 >= width2 || h2 >= height2) {
      return;
    }

    int nc_range = item.get_global_range(0);

    const size_t h1 =
        height1 == height2 ? h2 : index_op(height_scale, h2, height1);
    const size_t w1 = width1 == width2 ? w2 : index_op(width_scale, w2, width1);

    size_t src_index = (nc_idx * height1 + h1) * width1 + w1;
    size_t src_index_stride = nc_range * width1 * height1;
    size_t dst_index = (nc_idx * height2 + h2) * width2 + w2;
    size_t dst_index_stride = nc_range * width2 * height2;

    // iterating over
    while (nc_idx < nc) {
      odata[dst_index] = idata[src_index];
      dst_index += dst_index_stride;
      src_index += src_index_stride;
      nc_idx += nc_range;
    }
  }
  UpsampleNearest2dOutKernelFunctor(
      const scalar_t* idata_,
      scalar_t* odata_,
      const size_t nc_,
      const size_t height1_,
      const size_t width1_,
      const size_t height2_,
      const size_t width2_,
      float height_scale_,
      float width_scale_,
      index_op_t index_op_)
      : idata(idata_),
        odata(odata_),
        nc(nc_),
        height1(height1_),
        width1(width1_),
        height2(height2_),
        width2(width2_),
        height_scale(height_scale_),
        width_scale(width_scale_),
        index_op(index_op_) {}

 private:
  const scalar_t* idata;
  scalar_t* odata;
  const size_t nc;
  const size_t height1;
  const size_t width1;
  const size_t height2;
  const size_t width2;
  float height_scale;
  float width_scale;
  index_op_t index_op;
};

template <typename scalar_t, typename index_op_t>
void upsample_nearest2d_out_kernel(
    const scalar_t* idata,
    scalar_t* odata,
    const size_t nc,
    const size_t height1, // input height
    const size_t width1,
    const size_t height2, // output height
    const size_t width2,
    float height_scale,
    float width_scale,
    index_op_t index_op) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto work_group_size = dpcppMaxWorkItemsPerEU(dev_id);
  int local_x = std::min<int>(lastPow2(width2), work_group_size);
  int local_y = std::min<int>(lastPow2(height2), work_group_size / local_x);
  int local_z = std::min<int>(nc, work_group_size / local_x / local_y);

  int global_x = (width2 + local_x - 1) / local_x * local_x;
  int global_y = (height2 + local_y - 1) / local_y * local_y;
  int z_plane = local_z * 4;
  int global_z = (nc + z_plane - 1) / z_plane * z_plane;

  auto cgf = DPCPP_Q_CGF(cgh) {
    UpsampleNearest2dOutKernelFunctor<scalar_t, index_op_t> kfn(
        idata,
        odata,
        nc,
        height1,
        width1,
        height2,
        width2,
        height_scale,
        width_scale,
        index_op);

    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<3>(
            sycl::range<3>(global_z, global_y, global_x),
            sycl::range<3>(local_z, local_y, local_x)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename index_op_t>
struct UpsampleNearest2dChannelsLastKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    const int index = item.get_global_linear_id();

    if (index < out_numel) {
      const int c = index % channels;
      const int w2 = (index / channels) % width2;
      const int h2 = (index / channels / width2) % height2;
      const int n = index / channels / width2 / height2;

      const size_t h1 =
          height1 == height2 ? h2 : index_op(height_scale, h2, height1);
      const size_t w1 =
          width1 == width2 ? w2 : index_op(width_scale, w2, width1);

      odata[index] = idata[idx_cl(n, h1, w1, c, height1, width1, channels)];
    }
  }
  UpsampleNearest2dChannelsLastKernelFunctor(
      const scalar_t* idata_,
      scalar_t* odata_,
      const size_t channels_,
      const size_t height1_,
      const size_t width1_,
      const size_t height2_,
      const size_t width2_,
      float height_scale_,
      float width_scale_,
      const size_t out_numel_,
      index_op_t index_op_)
      : idata(idata_),
        odata(odata_),
        channels(channels_),
        height1(height1_),
        width1(width1_),
        height2(height2_),
        width2(width2_),
        height_scale(height_scale_),
        width_scale(width_scale_),
        out_numel(out_numel_),
        index_op(index_op_) {}

 private:
  const scalar_t* idata;
  scalar_t* odata;
  const size_t channels;
  const size_t height1;
  const size_t width1;
  const size_t height2;
  const size_t width2;
  float height_scale;
  float width_scale;
  const size_t out_numel;
  index_op_t index_op;
};

template <typename scalar_t, typename index_op_t>
void upsample_nearest2d_channels_last_kernel(
    const scalar_t* idata,
    scalar_t* odata,
    const size_t channels,
    const size_t height1,
    const size_t width1,
    const size_t height2,
    const size_t width2,
    float height_scale,
    float width_scale,
    const size_t out_numel,
    index_op_t index_op) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  auto work_group_size = dpcppMaxWorkItemsPerEU(dev_id);
  int global_range =
      (out_numel + work_group_size - 1) / work_group_size * work_group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    UpsampleNearest2dChannelsLastKernelFunctor<scalar_t, index_op_t> kfn(
        idata,
        odata,
        channels,
        height1,
        width1,
        height2,
        width2,
        height_scale,
        width_scale,
        out_numel,
        index_op);

    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(work_group_size)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

Tensor& _upsample_nearest_exact3d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& output) {
  torch_ipex::xpu::oneDNN::resample(
      input,
      output,
      output_size,
      algorithm::resampling_nearest,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f,
      scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0f);
  return output;
}

Tensor _upsample_nearest_exact3d(
    const Tensor& input,
    c10::OptionalIntArrayRef output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto output = at::empty({0}, input.options());
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  torch_ipex::xpu::oneDNN::resample(
      input,
      output,
      osize,
      algorithm::resampling_nearest,
      scale_w.has_value() ? static_cast<double>(scale_w.value()) : 0.0f,
      scale_h.has_value() ? static_cast<double>(scale_h.value()) : 0.0f,
      scale_d.has_value() ? static_cast<double>(scale_d.value()) : 0.0f);
  return output;
}

Tensor& _upsample_nearest_exact3d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  torch_ipex::xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      output_size,
      algorithm::resampling_nearest,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f,
      scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0f);
  return grad_input;
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
template <typename index_op_t>
static Tensor q_upsample_nearest2d_template(
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    index_op_t index_op) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_.dim() == 4,
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input_.sizes());

  if (input_.numel() == 0) {
    return input_;
  }

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_.size(0);
  int channels = input_.size(1);
  int input_height = input_.size(2);
  int input_width = input_.size(3);

  // TODO: upsample_nearest2d meta call makes sure input/output tensor is
  // not empty;
  //
  const float height_scale = nearest_compute_scales_value<float>(
      scales_h, input_height, output_height);
  const float width_scale =
      nearest_compute_scales_value<float>(scales_w, input_width, output_width);

  const auto memory_format = input_.suggest_memory_format();

  // heuristic: only use channels_last path when it's faster than the
  // contiguous path
  if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 4) {
    at::Tensor input = input_.contiguous(at::MemoryFormat::ChannelsLast);
    Tensor output = at::_empty_affine_quantized(
        {nbatch, channels, output_height, output_width},
        input.options()
            .memory_format(input.suggest_memory_format())
            .dtype(toQIntType(input.scalar_type())),
        input.q_scale(),
        input.q_zero_point());

    if (input_height == output_height && input_width == output_width) {
      output.copy_(input);
      return output;
    }

    TORCH_CHECK(
        input.numel() < std::numeric_limits<int>::max(),
        "upsample_nearest_channels_last only supports input tensors with less than INT_MAX elements");
    TORCH_CHECK(
        output.numel() < std::numeric_limits<int>::max(),
        "upsample_nearest_channels_last only supports output tensors with less than INT_MAX elements");

    IPEX_DISPATCH_QINT_TYPES(
        input.scalar_type(), "q_upsample_nearest2d_channels_last", [&]() {
          const auto idata =
              reinterpret_cast<underlying_t*>(input.data_ptr<scalar_t>());
          const auto odata =
              reinterpret_cast<underlying_t*>(output.data_ptr<scalar_t>());
          at::AtenIpexTypeXPU::upsample_nearest2d_channels_last_kernel<
              underlying_t>(
              idata,
              odata,
              channels,
              input_height,
              input_width,
              output_height,
              output_width,
              height_scale,
              width_scale,
              output.numel(),
              index_op);
        });
    return output;

  } else {
    // This is needed for non-contiguous tensors.
    Tensor input = input_.contiguous();

    Tensor output = at::_empty_affine_quantized(
        {nbatch, channels, output_height, output_width},
        input.options().dtype(toQIntType(input.scalar_type())),
        input.q_scale(),
        input.q_zero_point());
    Tensor output_c = output.is_contiguous()
        ? output
        : at::empty(output.sizes(), output.options());
    int nc = nbatch * channels;

    IPEX_DISPATCH_QINT_TYPES(
        input.scalar_type(), "q_upsample_nearest2d", [&]() {
          auto idata =
              reinterpret_cast<underlying_t*>(input.data_ptr<scalar_t>());
          auto odata =
              reinterpret_cast<underlying_t*>(output_c.data_ptr<scalar_t>());

          at::AtenIpexTypeXPU::upsample_nearest2d_out_kernel<underlying_t>(
              idata,
              odata,
              nc,
              input_height,
              input_width,
              output_height,
              output_width,
              height_scale,
              width_scale,
              index_op);
        });

    if (!output.is_contiguous()) {
      output.copy_(output_c);
    }
    return output;
  }
}

Tensor upsample_nearest2d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  int output_height = output_size[0];
  int output_width = output_size[1];

  int input_height = input.size(2);
  int input_width = input.size(3);
  bool onednn_path =
      (output_height % input_height == 0) && (output_width % input_width == 0);
  torch_ipex::xpu::COMPUTE_ENG real_eng;
  if (onednn_path) {
    real_eng = torch_ipex::xpu::COMPUTE_ENG::ONEDNN;
  } else {
    // Always use sycl implementation due to onednn path has acc issue
    real_eng = torch_ipex::xpu::COMPUTE_ENG::BASIC;
  }
  // temp fix: restore onednn path for integral scale cases
  // TODO: optimize perf for sycl implementation for both integral and
  // non-integral scale cases
  if (torch_ipex::xpu::COMPUTE_ENG::ONEDNN == real_eng) {
    Tensor output = at::_empty_affine_quantized(
        input.sizes(),
        input.options().dtype(toQIntType(input.scalar_type())),
        input.q_scale(),
        input.q_zero_point());
    torch_ipex::xpu::oneDNN::resample(
        input,
        output,
        output_size,
        algorithm::resampling_nearest,
        scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
        scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f);
    return output;
  } else {
    to_plain_if_needed_(input);

    return q_upsample_nearest2d_template(
        input,
        output_size,
        scales_h,
        scales_w,
        at::AtenIpexTypeXPU::Nearest_index_op());
  }
}

Tensor upsample_nearest3d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TORCH_INTERNAL_ASSERT(
      false,
      "upsample_nearest3d dosen't support quantized input, we will enable it in the future.");
}

Tensor upsample_nearest3d(
    const Tensor& input,
    at::OptionalIntArrayRef output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  TORCH_INTERNAL_ASSERT(
      false,
      "upsample_nearest3d dosen't support quantized input, we will enable it in the future.");
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
