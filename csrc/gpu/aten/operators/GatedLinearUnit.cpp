#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <runtime/Utils.h>
#ifdef USE_OVERRIDE_OP
#include <ATen/DeviceGuard.h>
#include <ATen/core/op_registration/adaption.h>
#include "utils/CustomOperatorRegistration.h"
#endif
#include <utils/DPCPP.h>

#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace torch_ipex::xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {
template <typename scalar_t>
void GatedLinearUnit_updateGradInput(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    int64_t dim) {
  TORCH_CHECK(input.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, input.dim());
  const int64_t nln = input.size(wrap_dim);
  TORCH_CHECK(
      nln % 2 == 0,
      "Halving dimension must be even, but dimension ",
      wrap_dim,
      " is size ",
      nln);

  grad_input.resize_as_(input);
  const int64_t inputSize = nln / 2;
  Tensor firstHalf = input.narrow(wrap_dim, 0, inputSize);
  Tensor secondHalf = input.narrow(wrap_dim, inputSize, inputSize);
  Tensor gradInputfirstHalf = grad_input.narrow(wrap_dim, 0, inputSize);
  Tensor gradInputsecondHalf =
      grad_input.narrow(wrap_dim, inputSize, inputSize);

  // gradInputfirstHalf = grad_output * sigmoid(secondHalf)
  // gradInputsecondHalf = (1 - sigmoid(secondHalf)) * sigmoid(secondHalf) *
  // input * grad_output
  at::sigmoid_out(gradInputfirstHalf, secondHalf);
  gradInputsecondHalf.fill_(ScalarConvert<int, scalar_t>::to(1));
  gradInputsecondHalf.sub_(gradInputfirstHalf)
      .mul_(gradInputfirstHalf)
      .mul_(firstHalf);
  gradInputfirstHalf.mul_(grad_output);
  gradInputsecondHalf.mul_(grad_output);
}

} // namespace impl

// namespace AtenIpexTypeXPU
Tensor& glu_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    int64_t dim,
    Tensor& grad_input) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "glu_backward_out",
      [&] {
        impl::GatedLinearUnit_updateGradInput<scalar_t>(
            grad_input, grad_output, self, dim);
      });
  return grad_input;
}

Tensor glu_backward(
    const Tensor& grad_output,
    const Tensor& self,
    int64_t dim) {
  Tensor grad_input = at::empty({}, self.options());
  return at::AtenIpexTypeXPU::glu_backward_out(
      grad_output, self, dim, grad_input);
}

} // namespace AtenIpexTypeXPU
} // namespace at

#ifdef USE_OVERRIDE_OP
namespace {
at::Tensor wrapper_XPU__glu_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    int64_t dim) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, grad_output, "wrapper_XPU__glu_backward", "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU__glu_backward", "self");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::glu_backward(grad_output, self, dim);
}

at::Tensor& wrapper_XPU_grad_input_glu_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    int64_t dim,
    at::Tensor& grad_input) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device,
      grad_input,
      "wrapper_XPU_grad_input_glu_backward_out",
      "grad_input");
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "wrapper_XPU_grad_input_glu_backward_out",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_XPU_grad_input_glu_backward_out", "self");
  const OptionalDeviceGuard device_guard(device_of(self));

  return at::AtenIpexTypeXPU::glu_backward_out(
      grad_output, self, dim, grad_input);
}

IPEX_TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("glu_backward", TORCH_FN(&wrapper_XPU__glu_backward));
  m.impl(
      "glu_backward.grad_input",
      TORCH_FN(&wrapper_XPU_grad_input_glu_backward_out));
}

} // namespace
#endif
