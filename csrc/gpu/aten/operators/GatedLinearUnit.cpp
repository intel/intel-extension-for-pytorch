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

template <typename scalar_t, typename opmath_t>
struct glu_jvp_functor {
  scalar_t operator()(scalar_t res_, scalar_t b_, scalar_t da_, scalar_t db_)
      const {
    const opmath_t res = res_;
    const opmath_t b = b_;
    const opmath_t da = da_;
    const opmath_t db = db_;
    const opmath_t one = opmath_t(1.0f);

    const opmath_t sig_b = one / (one + Numerics<opmath_t>::exp(-b));
    return (da * sig_b + res * (db - sig_b * db));
  }
};

Tensor glu_jvp(
    const Tensor& glu,
    const Tensor& x,
    const Tensor& dx,
    int64_t dim) {
  dim = maybe_wrap_dim(dim, x.dim());
  const auto glu_size = glu.size(dim);
  const auto b = x.narrow(dim, glu_size, glu_size);
  const auto da = dx.narrow(dim, 0, glu_size);
  const auto db = dx.narrow(dim, glu_size, glu_size);
  auto dglu = at::empty_like(glu);
  auto iter = at::TensorIteratorConfig()
                  .add_output(dglu)
                  .add_input(glu)
                  .add_input(b)
                  .add_input(da)
                  .add_input(db)
                  .build();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.dtype(), "glu_jvp", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        glu_jvp_functor<scalar_t, opmath_t> f;
        dpcpp_kernel_for_tensor_iter(iter, f);
      });
  return dglu;
}

Tensor glu_backward_jvp(
    const Tensor& grad_x,
    const Tensor& grad_glu,
    const Tensor& x,
    const Tensor& dgrad_glu,
    const Tensor& dx,
    int64_t dim) {
  dim = maybe_wrap_dim(dim, x.dim());
  const auto glu_size = grad_glu.size(dim);
  const auto a = x.narrow(dim, 0, glu_size);
  const auto b = x.narrow(dim, glu_size, glu_size);
  const auto da = dx.narrow(dim, 0, glu_size);
  const auto db = dx.narrow(dim, glu_size, glu_size);
  // grad_x_a = grad_glu * sigmoid(b)
  const auto grad_x_a = grad_x.narrow(dim, 0, glu_size);
  // grad_x_b = grad_x_a * a * (1 - sigmoid(b))
  const auto grad_x_b = grad_x.narrow(dim, glu_size, glu_size);

  const auto sig_b = at::sigmoid(b);
  // TODO: use glu from forward.
  // TODO: fuse kernels.
  const auto glu = a * sig_b;
  const auto db_neg_sig_b = db - db * sig_b;

  // dgrad_x_a = d(grad_glu * sigmoid(b))
  //           = dgrad_glu * sigmoid(b) + grad_glu * sigmoid(b) * (1 -
  //           sigmoid(b)) * db = dgrad_glu * sig_b + grad_x_a * (db - db *
  //           sig_b) = dgrad_glu * sig_b + grad_x_a * db_neg_sig_b
  const auto dgrad_x_a = dgrad_glu * sig_b + grad_x_a * db_neg_sig_b;

  // dgrad_x_b = d(grad_glu * sigmoid(b) * a * (1 - sigmoid(b))
  //           =  d(grad_glu * sigmoid(b)) * a * (1 - sigmoid(b))
  //            + grad_glu * sigmoid(b) * da * (1 - sigmoid(b))
  //            - grad_glu * sigmoid(b) * a * sigmoid(b) * (1 - sigmoid(b)) * db
  //          =   dgrad_x_a * a * (1 - sigmoid(b))
  //           + (grad_glu * sigmoid(b)) * (da * (1 - sigmoid(b)) - a *
  //           sigmoid(b) * (1 - sigmoid(b)) * db)
  //          = dgrad_x_a * (a - glu) + grad_x_a * (da - da * sig_b - glu *
  //          db_neg_sig_b
  const auto dgrad_x_b =
      dgrad_x_a * (a - glu) + grad_x_a * (da - da * sig_b - glu * db_neg_sig_b);
  auto out = at::cat({dgrad_x_a, dgrad_x_b}, dim);

  return at::cat({dgrad_x_a, dgrad_x_b}, dim);
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
