#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NativeFunctions.h>
#include <ATen/record_function.h>
#include <grp.h>
#include <oneapi/dnnl/dnnl.hpp>

#include <oneDNN/oneDNN.h>

using namespace xpu::oneDNN;

namespace torch {
namespace jit {
namespace xpu {

at::Tensor dequant_pixelshuffle(
    const at::Tensor& self,
    int64_t upscale_factor) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::empty_like(self);
}

at::Tensor dequant_pixelshuffle_quant(
    const at::Tensor& self,
    int64_t upscale_factor,
    double scale,
    int64_t zero_pad,
    at::ScalarType dtype) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::pixel_shuffle(self, upscale_factor);
}

at::Tensor batch_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double momentum,
    double eps,
    bool use_dnn) {
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::empty_like(input);
}

at::Tensor fold_weight(
    const at::Tensor& weight,
    const at::Tensor& bn_weight,
    const at::Tensor& running_var,
    float eps) {
  const OptionalDeviceGuard device_guard(device_of(weight));
  return at::empty_like(weight);
}

at::Tensor fold_bias(
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& bn_weight,
    const at::Tensor& bn_bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    float eps) {
  const OptionalDeviceGuard device_guard(device_of(weight));
  return at::empty_like(bias);
}

at::Tensor reorder(
    const at::Tensor& input,
    dnnl::memory::format_tag from,
    dnnl::memory::format_tag to,
    int64_t groups) {
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::empty_like(input);
}

} // namespace xpu

} // namespace jit
} // namespace torch
