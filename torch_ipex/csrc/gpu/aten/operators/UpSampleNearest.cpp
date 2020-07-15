#include "UpSample.h"

using namespace dnnl;
using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

static void upsample_nearest_out_dpcpp_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    const double& scales_w = 0.0,
    const double& scales_h = 0.0,
    const double& scales_d = 0.0) {
  auto input = input_.contiguous();

  auto strm = GpuStreamManager::Instance().get_stream();
  Device curDevice = Device(kDPCPP, current_device());
  auto eng = GpuEngineManager::Instance().get_engine(curDevice);

  bool is_customer_scales =
      scales_w != 0.0 || scales_h != 0.0 || scales_d != 0.0;

  int64_t ndims = input.ndimension();
  IntArrayRef input_size = input.sizes();
  memory::dims src_dims, dst_dims;
  std::vector<float> factors;
  set_params(
      input_size,
      output_size,
      src_dims,
      dst_dims,
      factors,
      ndims,
      scales_w,
      scales_h,
      scales_d);

  output.resize_(dst_dims);
  output.zero_();

  memory::format_tag data_format = ndims == 5
      ? memory::format_tag::ncdhw
      : (ndims == 4 ? memory::format_tag::nchw : memory::format_tag::ncw);
  memory::data_type data_type = dt_to_dnnl(input.scalar_type());

  std::shared_ptr<memory::desc> src_desc, dst_desc;
  src_desc.reset(new memory::desc(src_dims, data_type, data_format));
  if (!is_customer_scales)
    dst_desc.reset(new memory::desc(dst_dims, data_type, data_format));

  auto resampling_desc = resampling_forward::desc(
      prop_kind::forward,
      algorithm::resampling_nearest,
      factors,
      *src_desc,
      *dst_desc);
  auto resampling_pd = resampling_forward::primitive_desc(resampling_desc, eng);
  resampling_pd = resampling_forward::primitive_desc(resampling_pd.get());

  auto src = dpcpp_onednn_memory(
      resampling_pd.src_desc(), eng, input.data_ptr());
  auto dst = dpcpp_onednn_memory(
      resampling_pd.dst_desc(), eng, output.data_ptr());

  DPCPP_ONEDNN_EXEC(resampling_forward(resampling_pd),
      strm, {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
}

static void upsample_nearest_backward_out_dpcpp_kernel(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    const double& scales_w = 0.0,
    const double& scales_h = 0.0,
    const double& scales_d = 0.0) {
  auto grad_output = grad_output_.contiguous();

  auto strm = GpuStreamManager::Instance().get_stream();
  Device curDevice = Device(kDPCPP, current_device());
  auto eng = GpuEngineManager::Instance().get_engine(curDevice);

  bool is_customer_scales =
      scales_w != 0.0 || scales_h != 0.0 || scales_d != 0.0;

  int64_t ndims = grad_output.ndimension();
  memory::dims src_dims, dst_dims;
  std::vector<float> factors;
  set_params(
      input_size,
      output_size,
      src_dims,
      dst_dims,
      factors,
      ndims,
      scales_w,
      scales_h,
      scales_d);

  grad_input.resize_(src_dims);
  grad_input.zero_();

  memory::format_tag data_format = ndims == 5
      ? memory::format_tag::ncdhw
      : (ndims == 4 ? memory::format_tag::nchw : memory::format_tag::ncw);
  memory::data_type data_type = dt_to_dnnl(grad_output.scalar_type());

  std::shared_ptr<memory::desc> src_desc, dst_desc;
  src_desc.reset(new memory::desc(src_dims, data_type, data_format));
  if (!is_customer_scales)
    dst_desc.reset(new memory::desc(dst_dims, data_type, data_format));

  auto resampling_desc = resampling_forward::desc(
      prop_kind::forward,
      algorithm::resampling_nearest,
      factors,
      *src_desc,
      *dst_desc);
  auto resampling_pd = resampling_forward::primitive_desc(resampling_desc, eng);
  resampling_pd = resampling_forward::primitive_desc(resampling_pd.get());

  auto resampling_bwd_desc = resampling_backward::desc(
      algorithm::resampling_nearest,
      factors,
      *src_desc,
      resampling_pd.dst_desc());
  auto resampling_bwd_pd = resampling_backward::primitive_desc(
      resampling_bwd_desc, eng, resampling_pd);
  resampling_bwd_pd =
      resampling_backward::primitive_desc(resampling_bwd_pd.get());

  memory grad_src = dpcpp_onednn_memory(
      resampling_bwd_pd.diff_src_desc(), eng, grad_input.data_ptr());
  memory grad_dst = dpcpp_onednn_memory(
      resampling_bwd_pd.diff_dst_desc(), eng, grad_output.data_ptr());

  DPCPP_ONEDNN_EXEC(resampling_backward(resampling_bwd_pd),
      strm, {{DNNL_ARG_DIFF_SRC, grad_src}, {DNNL_ARG_DIFF_DST, grad_dst}});
}

} // namespace impl

using namespace impl;

Tensor& upsample_nearest3d_out(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  upsample_nearest_out_dpcpp_kernel(
      output,
      input,
      output_size,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0,
      scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0);
  return output;
}

Tensor upsample_nearest3d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto output = at::empty({0}, input.options());
  upsample_nearest_out_dpcpp_kernel(
      output,
      input,
      output_size,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0,
      scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0);
  return output;
}

Tensor& upsample_nearest3d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  upsample_nearest_backward_out_dpcpp_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0,
      scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0);
  return grad_input;
}

Tensor upsample_nearest3d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto grad_input = at::empty({0}, grad_output.options());
  upsample_nearest_backward_out_dpcpp_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0,
      scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0);
  return grad_input;
}

Tensor& upsample_nearest2d_out(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  upsample_nearest_out_dpcpp_kernel(
      output,
      input,
      output_size,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0);
  return output;
}

Tensor upsample_nearest2d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto output = at::empty({0}, input.options());
  upsample_nearest_out_dpcpp_kernel(
      output,
      input,
      output_size,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0);
  return output;
}

Tensor& upsample_nearest2d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  upsample_nearest_backward_out_dpcpp_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0);
  return grad_input;
}

Tensor upsample_nearest2d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto grad_input = at::empty({0}, grad_output.options());
  upsample_nearest_backward_out_dpcpp_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0);
  return grad_input;
}

Tensor& upsample_nearest1d_out(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales) {
  upsample_nearest_out_dpcpp_kernel(
      output,
      input,
      output_size,
      scales.has_value() ? static_cast<double>(scales.value()) : 0.0);
  return output;
}

Tensor upsample_nearest1d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales) {
  auto output = at::empty({0}, input.options());
  upsample_nearest_out_dpcpp_kernel(
      output,
      input,
      output_size,
      scales.has_value() ? static_cast<double>(scales.value()) : 0.0);
  return output;
}

Tensor& upsample_nearest1d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales) {
  upsample_nearest_backward_out_dpcpp_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales.has_value() ? static_cast<double>(scales.value()) : 0.0);
  return grad_input;
}

Tensor upsample_nearest1d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales) {
  auto grad_input = at::zeros(input_size, grad_output.options());
  upsample_nearest_backward_out_dpcpp_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales.has_value() ? static_cast<double>(scales.value()) : 0.0);
  return grad_input;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
