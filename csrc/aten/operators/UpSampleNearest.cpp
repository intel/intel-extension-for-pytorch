#include <ATen/native/UpSample.h>
#include <tensor/Tensor.h>
#include "UpSample.h"
#include "comm/RegistrationDeclarations.h"

#include <oneDNN/oneDNN.h>

using namespace dnnl;
using namespace at::native;
using namespace at::AtenIpexTypeXPU;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {

using namespace impl;
using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

Tensor& upsample_nearest3d_out(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  xpu::oneDNN::resample(
      input,
      output,
      output_size,
      algorithm::resampling_nearest,
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
  xpu::oneDNN::resample(
      input,
      output,
      output_size,
      algorithm::resampling_nearest,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0,
      scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0);
  return output;
}

Tensor upsample_nearest3d(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto output = at::empty({0}, input.options());
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  xpu::oneDNN::resample(
      input,
      output,
      osize,
      algorithm::resampling_nearest,
      scale_w.has_value() ? static_cast<double>(scale_w.value()) : 0.0,
      scale_h.has_value() ? static_cast<double>(scale_h.value()) : 0.0,
      scale_d.has_value() ? static_cast<double>(scale_d.value()) : 0.0);
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
  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      output_size,
      algorithm::resampling_nearest,
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
  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      output_size,
      algorithm::resampling_nearest,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0,
      scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0);
  return grad_input;
}

Tensor upsample_nearest3d_backward(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input_size, output_size, scale_factors);
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  auto grad_input = at::empty({0}, grad_output.options());
  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      osize,
      algorithm::resampling_nearest,
      scale_w.has_value() ? static_cast<double>(scale_w.value()) : 0.0,
      scale_h.has_value() ? static_cast<double>(scale_h.value()) : 0.0,
      scale_d.has_value() ? static_cast<double>(scale_d.value()) : 0.0);
  return grad_input;
}

Tensor& upsample_nearest2d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& output) {
  xpu::oneDNN::resample(
      input,
      output,
      output_size,
      algorithm::resampling_nearest,
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
  xpu::oneDNN::resample(
      input,
      output,
      output_size,
      algorithm::resampling_nearest,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0);
  return output;
}

Tensor upsample_nearest2d(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  xpu::oneDNN::resample(
      input,
      output,
      osize,
      algorithm::resampling_nearest,
      scale_w.has_value() ? static_cast<double>(scale_w.value()) : 0.0,
      scale_h.has_value() ? static_cast<double>(scale_h.value()) : 0.0);
  return output;
}

Tensor& upsample_nearest2d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      output_size,
      algorithm::resampling_nearest,
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
  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      output_size,
      algorithm::resampling_nearest,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0);
  return grad_input;
}

Tensor upsample_nearest2d_backward(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input_size, output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  auto grad_input = at::empty({0}, grad_output.options());
  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      osize,
      algorithm::resampling_nearest,
      scale_w.has_value() ? static_cast<double>(scale_w.value()) : 0.0,
      scale_h.has_value() ? static_cast<double>(scale_h.value()) : 0.0);
  return grad_input;
}

Tensor& upsample_nearest1d_out(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales) {
  xpu::oneDNN::resample(
      input,
      output,
      output_size,
      algorithm::resampling_nearest,
      scales.has_value() ? static_cast<double>(scales.value()) : 0.0);
  return output;
}

Tensor upsample_nearest1d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales) {
  auto output = at::empty({0}, input.options());
  xpu::oneDNN::resample(
      input,
      output,
      output_size,
      algorithm::resampling_nearest,
      scales.has_value() ? static_cast<double>(scales.value()) : 0.0);
  return output;
}

Tensor upsample_nearest1d(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_w = get_scale_value(scale_factors, 0);
  xpu::oneDNN::resample(
      input,
      output,
      osize,
      algorithm::resampling_nearest,
      scale_w.has_value() ? static_cast<double>(scale_w.value()) : 0.0);
  return output;
}

Tensor& upsample_nearest1d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales) {
  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      output_size,
      algorithm::resampling_nearest,
      scales.has_value() ? static_cast<double>(scales.value()) : 0.0);
  return grad_input;
}

Tensor upsample_nearest1d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales) {
  auto grad_input = at::empty({0}, grad_output.options());
  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      output_size,
      algorithm::resampling_nearest,
      scales.has_value() ? static_cast<double>(scales.value()) : 0.0);
  return grad_input;
}

Tensor upsample_nearest1d_backward(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input_size, output_size, scale_factors);
  auto scale_w = get_scale_value(scale_factors, 0);
  auto grad_input = at::empty({0}, grad_output.options());
  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      osize,
      algorithm::resampling_nearest,
      scale_w.has_value() ? static_cast<double>(scale_w.value()) : 0.0);
  return grad_input;
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
Tensor upsample_nearest2d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  Tensor output = at::_empty_affine_quantized(
      input.sizes(),
      input.options().dtype(toQIntType(input.scalar_type())),
      input.q_scale(),
      input.q_zero_point());
  xpu::oneDNN::resample(
      input,
      output,
      output_size,
      algorithm::resampling_nearest,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0);
  return output;
}

Tensor upsample_nearest2d(
    const Tensor& input,
    OptionalIntArrayRef output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  Tensor output = at::_empty_affine_quantized(
      input.sizes(),
      input.options().dtype(toQIntType(input.scalar_type())),
      input.q_scale(),
      input.q_zero_point());
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  xpu::oneDNN::resample(
      input,
      output,
      osize,
      algorithm::resampling_nearest,
      scale_w.has_value() ? static_cast<double>(scale_w.value()) : 0.0,
      scale_h.has_value() ? static_cast<double>(scale_h.value()) : 0.0);
  return output;
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
