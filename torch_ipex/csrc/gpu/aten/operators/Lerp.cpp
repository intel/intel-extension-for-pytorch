#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <core/ApplyUtils.h>
#include <utils/Numerics.h>
#include <utils/ATDispatch.h>

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename T>
struct TensorLerpScaleOp {
  TensorLerpScaleOp(T w) : w(w) {}

  void operator()(T& out, T& a, T& b) const {
    out = (w < 0.5)
        ? Numerics<T>::add(a, Numerics<T>::mul(w, Numerics<T>::sub(b, a)))
        : Numerics<T>::sub(
              b,
              Numerics<T>::mul(Numerics<T>::sub(b, a), Numerics<T>::sub(1, w)));
  }

  const T w;
};

template <typename T>
struct TensorLerpOp {
  void operator()(T& out, T& a, T& b, T& weight) const {
    out = (weight < 0.5)
        ? Numerics<T>::add(a, Numerics<T>::mul(weight, Numerics<T>::sub(b, a)))
        : Numerics<T>::sub(
              b,
              Numerics<T>::mul(
                  Numerics<T>::sub(b, a), Numerics<T>::sub(1, weight)));
  }
};

template <typename scalar_t>
void lerp(
    at::Tensor& ret,
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Tensor& weight) {
  DPCPP_tensor_apply4<scalar_t, scalar_t, scalar_t, scalar_t>(
      ret, self, end, weight, TensorLerpOp<scalar_t>());
}

template <typename scalar_t>
void lerp(
    at::Tensor& ret,
    const at::Tensor& self,
    const at::Tensor& end,
    scalar_t weight_val) {
  DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      ret, self, end, TensorLerpScaleOp<scalar_t>(weight_val));
}

} // namespace impl

Tensor& lerp_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& end,
    const Tensor& weight) {
  Tensor b_self, b_end, b_weight;
  TORCH_CHECK(
      weight.dim() <= std::max(self.dim(), end.dim()),
      "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  std::tie(b_self, b_end, b_weight) =
      expand_outplace(self, end, weight, "lerp_out");
  out.resize_as_(b_self);
  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lerp_out", [&] {
    impl::lerp<scalar_t>(out, b_self, b_end, b_weight);
  });
  return out;
}

Tensor& lerp_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& end,
    Scalar weight) {
  Tensor b_self, b_end;
  std::tie(b_self, b_end) = expand_outplace(self, end, "lerp_out");
  out.resize_as_(b_self);
  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lerp_out", [&] {
    impl::lerp<scalar_t>(out, b_self, b_end, weight.to<scalar_t>());
  });
  return out;
}

Tensor& lerp_(Tensor& self, const Tensor& end, const Tensor& weight) {
  Tensor b_self, b_end, b_weight;
  std::tie(b_self, b_end, b_weight) =
      expand_outplace(self, end, weight, "lerp_");
  TORCH_CHECK(
      b_self.sizes() == self.sizes(),
      "output with shape ",
      self.sizes(),
      " doesn't match the broadcast shape ",
      b_self.sizes());
  TORCH_CHECK(
      weight.dim() <= std::max(self.dim(), end.dim()),
      "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lerp_", [&] {
    impl::lerp<scalar_t>(self, b_self, b_end, b_weight);
  });
  return self;
}

Tensor& lerp_(Tensor& self, const Tensor& end, Scalar weight) {
  Tensor b_self, b_end;
  std::tie(b_self, b_end) = expand_outplace(self, end, "lerp_");
  TORCH_CHECK(
      b_self.sizes() == self.sizes(),
      "output with shape ",
      self.sizes(),
      " doesn't match the broadcast shape ",
      b_self.sizes());
  IPEX_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lerp_", [&] {
    impl::lerp<scalar_t>(self, b_self, b_end, weight.to<scalar_t>());
  });
  return self;
}

Tensor lerp(const Tensor& self, const Tensor& end, const Tensor& weight) {
  Tensor result = at::empty_like(self);
  return at::AtenIpexTypeXPU::lerp_out(result, self, end, weight);
}

Tensor lerp(const Tensor& self, const Tensor& end, Scalar weight) {
  Tensor result = at::empty_like(self);
  return at::AtenIpexTypeXPU::lerp_out(result, self, end, weight);
}

} // namespace AtenIpexTypeXPU
} // namespace at
