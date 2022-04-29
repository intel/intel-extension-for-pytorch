#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
void lerp(
    at::Tensor& ret,
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Tensor& weight) {
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(ret)
                  .add_input(self)
                  .add_input(end)
                  .add_input(weight)
                  .build();
  dpcpp_kernel_for_tensor_iter(
      iter, [=](scalar_t a, scalar_t b, scalar_t weight) -> scalar_t {
        return (weight < 0.5) ? Numerics<scalar_t>::add(
                                    a,
                                    Numerics<scalar_t>::mul(
                                        weight, Numerics<scalar_t>::sub(b, a)))
                              : Numerics<scalar_t>::sub(
                                    b,
                                    Numerics<scalar_t>::mul(
                                        Numerics<scalar_t>::sub(b, a),
                                        Numerics<scalar_t>::sub(1, weight)));
      });
}

} // namespace impl

Tensor& lerp_out(
    const Tensor& self,
    const Tensor& end,
    const Tensor& weight,
    Tensor& out) {
  c10::MaybeOwned<Tensor> b_self, b_end, b_weight;
  TORCH_CHECK(
      weight.dim() <= std::max(self.dim(), end.dim()),
      "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  std::tie(b_self, b_end, b_weight) =
      expand_outplace(self, end, weight, "lerp_out");
  out.resize_as_(*b_self);
  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lerp_out", [&] {
    impl::lerp<scalar_t>(out, *b_self, *b_end, *b_weight);
  });
  return out;
}

Tensor& lerp_out(
    const Tensor& self,
    const Tensor& end,
    const Scalar& weight,
    Tensor& out) {
  c10::MaybeOwned<Tensor> b_self, b_end;
  std::tie(b_self, b_end) = expand_outplace(self, end, "lerp_out");
  out.resize_as_(*b_self);
  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lerp_out", [&] {
    impl::lerp<scalar_t>(
        out,
        *b_self,
        *b_end,
        wrapped_scalar_tensor(weight, at::kXPU).to(self.dtype()));
  });
  return out;
}

Tensor& lerp_(Tensor& self, const Tensor& end, const Tensor& weight) {
  c10::MaybeOwned<Tensor> b_self, b_end, b_weight;
  std::tie(b_self, b_end, b_weight) =
      expand_outplace(self, end, weight, "lerp_");
  TORCH_CHECK(
      (*b_self).sizes() == self.sizes(),
      "output with shape ",
      self.sizes(),
      " doesn't match the broadcast shape ",
      (*b_self).sizes());
  TORCH_CHECK(
      weight.dim() <= std::max(self.dim(), end.dim()),
      "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lerp_", [&] {
    impl::lerp<scalar_t>(self, *b_self, *b_end, *b_weight);
  });
  return self;
}

Tensor& lerp_(Tensor& self, const Tensor& end, const Scalar& weight) {
  c10::MaybeOwned<Tensor> b_self, b_end;
  std::tie(b_self, b_end) = expand_outplace(self, end, "lerp_");
  TORCH_CHECK(
      (*b_self).sizes() == self.sizes(),
      "output with shape ",
      self.sizes(),
      " doesn't match the broadcast shape ",
      (*b_self).sizes());
  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lerp_", [&] {
    impl::lerp<scalar_t>(
        self,
        *b_self,
        *b_end,
        wrapped_scalar_tensor(weight, kXPU).to(self.dtype()));
  });
  return self;
}

Tensor lerp(const Tensor& self, const Tensor& end, const Tensor& weight) {
  Tensor result = at::empty_like(self);
  return at::AtenIpexTypeXPU::lerp_out(self, end, weight, result);
}

Tensor lerp(const Tensor& self, const Tensor& end, const Scalar& weight) {
  Tensor result = at::empty_like(self);
  return at::AtenIpexTypeXPU::lerp_out(
      self, end, wrapped_scalar_tensor(weight, kXPU).to(self.dtype()), result);
}

} // namespace AtenIpexTypeXPU
} // namespace at
