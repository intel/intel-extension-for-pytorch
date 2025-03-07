#include <ATen/AccumulateType.h>
#include <ATen/DeviceGuard.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include "comm/xpu_aten.h"
#include "sycl/BatchNormKernels.h"

#ifdef USE_OVERRIDE_OP
#include "utils/CustomOperatorRegistration.h"
#endif

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> batch_norm_xpu(
    const Tensor& input,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    bool training,
    double momentum,
    double eps) {
  auto output = at::empty_like(input);
  int64_t n_input = input.size(1);
  auto options =
      input.options().dtype(at::toAccumulateType(input.scalar_type(), true));
  auto save_mean = at::empty({n_input}, options);
  auto save_invstd = at::empty({n_input}, options);
  xpu::batch_norm_kernel(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      training,
      momentum,
      eps,
      output,
      save_mean,
      save_invstd);
  return std::make_tuple(output, save_mean, save_invstd);
}

std::tuple<Tensor&, Tensor&, Tensor&> batch_norm_xpu_out(
    const Tensor& input,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    bool training,
    double momentum,
    double eps,
    Tensor& out,
    Tensor& save_mean,
    Tensor& save_invstd) {
  return xpu::batch_norm_kernel(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      training,
      momentum,
      eps,
      out,
      save_mean,
      save_invstd);
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_xpu(
    const Tensor& grad_out,
    const Tensor& input,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    const std::optional<Tensor>& save_mean,
    const std::optional<Tensor>& save_invstd,
    bool train,
    double eps,
    std::array<bool, 3> output_mask) {
  return xpu::batch_norm_backward_kernel(
      grad_out,
      input,
      weight,
      running_mean,
      running_var,
      save_mean,
      save_invstd,
      train,
      eps,
      output_mask);
}
} // namespace native
} // namespace at

#ifdef USE_OVERRIDE_OP
namespace {

::std::tuple<at::Tensor, at::Tensor, at::Tensor> wrapper_XPU__native_batch_norm(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    bool training,
    double momentum,
    double eps) {
  c10::optional<c10::Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, input, "wrapper_XPU__native_batch_norm", "input");
  c10::impl::check_and_update_common_device(
      common_device, weight, "wrapper_XPU__native_batch_norm", "weight");
  c10::impl::check_and_update_common_device(
      common_device, bias, "wrapper_XPU__native_batch_norm", "bias");
  c10::impl::check_and_update_common_device(
      common_device,
      running_mean,
      "wrapper_XPU__native_batch_norm",
      "running_mean");
  c10::impl::check_and_update_common_device(
      common_device,
      running_var,
      "wrapper_XPU__native_batch_norm",
      "running_var");
  const c10::OptionalDeviceGuard device_guard(device_of(input));

  return at::native::batch_norm_xpu(
      input, weight, bias, running_mean, running_var, training, momentum, eps);
}

::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>
wrapper_XPU_out_native_batch_norm_out(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    bool training,
    double momentum,
    double eps,
    at::Tensor& out,
    at::Tensor& save_mean,
    at::Tensor& save_invstd) {
  c10::optional<c10::Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_XPU_out_native_batch_norm_out", "out");
  c10::impl::check_and_update_common_device(
      common_device,
      save_mean,
      "wrapper_XPU_out_native_batch_norm_out",
      "save_mean");
  c10::impl::check_and_update_common_device(
      common_device,
      save_invstd,
      "wrapper_XPU_out_native_batch_norm_out",
      "save_invstd");
  c10::impl::check_and_update_common_device(
      common_device, input, "wrapper_XPU_out_native_batch_norm_out", "input");
  c10::impl::check_and_update_common_device(
      common_device, weight, "wrapper_XPU_out_native_batch_norm_out", "weight");
  c10::impl::check_and_update_common_device(
      common_device, bias, "wrapper_XPU_out_native_batch_norm_out", "bias");
  c10::impl::check_and_update_common_device(
      common_device,
      running_mean,
      "wrapper_XPU_out_native_batch_norm_out",
      "running_mean");
  c10::impl::check_and_update_common_device(
      common_device,
      running_var,
      "wrapper_XPU_out_native_batch_norm_out",
      "running_var");
  const c10::OptionalDeviceGuard device_guard(device_of(out));

  return at::native::batch_norm_xpu_out(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      training,
      momentum,
      eps,
      out,
      save_mean,
      save_invstd);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor>
wrapper_XPU__native_batch_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    const c10::optional<at::Tensor>& save_mean,
    const c10::optional<at::Tensor>& save_invstd,
    bool train,
    double eps,
    ::std::array<bool, 3> output_mask) {
  c10::optional<c10::Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device,
      grad_out,
      "wrapper_XPU__native_batch_norm_backward",
      "grad_out");
  c10::impl::check_and_update_common_device(
      common_device, input, "wrapper_XPU__native_batch_norm_backward", "input");
  c10::impl::check_and_update_common_device(
      common_device,
      weight,
      "wrapper_XPU__native_batch_norm_backward",
      "weight");
  c10::impl::check_and_update_common_device(
      common_device,
      running_mean,
      "wrapper_XPU__native_batch_norm_backward",
      "running_mean");
  c10::impl::check_and_update_common_device(
      common_device,
      running_var,
      "wrapper_XPU__native_batch_norm_backward",
      "running_var");
  c10::impl::check_and_update_common_device(
      common_device,
      save_mean,
      "wrapper_XPU__native_batch_norm_backward",
      "save_mean");
  c10::impl::check_and_update_common_device(
      common_device,
      save_invstd,
      "wrapper_XPU__native_batch_norm_backward",
      "save_invstd");
  const c10::OptionalDeviceGuard device_guard(device_of(grad_out));

  return at::native::batch_norm_backward_xpu(
      grad_out,
      input,
      weight,
      running_mean,
      running_var,
      save_mean,
      save_invstd,
      train,
      eps,
      output_mask);
}
// IPEX_TORCH_LIBRARY_IMPL(aten, XPU, m) {
//   m.impl("native_batch_norm", TORCH_FN((&wrapper_XPU__native_batch_norm)));
//   m.impl(
//       "native_batch_norm.out",
//       TORCH_FN((&wrapper_XPU_out_native_batch_norm_out)));
//   m.impl(
//       "native_batch_norm_backward",
//       TORCH_FN((&wrapper_XPU__native_batch_norm_backward)));
// }
} // namespace
#endif
