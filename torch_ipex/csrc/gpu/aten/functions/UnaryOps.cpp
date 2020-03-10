#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <core/SYCL.h>
#include <utils/Numerics.h>
#include <utils/Pointwise.h>
#include <utils/Pairwise.h>
#include <functions/Loops.h>


using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

DP_DEF_K1(bitwise_not);
void bitwise_not_kernel_sycl(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    sycl_kernel_for_tensor_iter<DP_K(bitwise_not)>(iter, [](bool a) -> bool {
      return !a;
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_sycl", [&]() {
      sycl_kernel_for_tensor_iter<DP_K(bitwise_not)>(iter, [](scalar_t a) -> scalar_t {
        return ~a;
      });
    });
  }
}

DP_DEF_K1(logical_not);
void logical_not_kernel_sycl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(1), "logical_not_sycl", [&]() {
    using self_t = scalar_t;
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(0), "logical_not_sycl", [&]() {
      sycl_kernel_for_tensor_iter<DP_K(logical_not, self_t)>(iter, [](self_t a) -> scalar_t { return static_cast<scalar_t>(!a); });
    });
  });
}

} // namespace impl

IPEX_OUT_ALL_UNARY_FUNC_OPS(abs_out, Numerics<scalar_t>::abs, Real);
IPEX_OUT_ALL_UNARY_FUNC_OPS(neg_out, Numerics<scalar_t>::neg, Real);

IPEX_OUT_FLOAT_UNARY_FUNC_OPS(cos_out, Numerics<scalar_t>::cos, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(sin_out, Numerics<scalar_t>::sin, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(cosh_out, Numerics<scalar_t>::cosh, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(sinh_out, Numerics<scalar_t>::sinh, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(acos_out, Numerics<scalar_t>::acos, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(asin_out, Numerics<scalar_t>::asin, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(floor_out, Numerics<scalar_t>::floor, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(expm1_out, Numerics<scalar_t>::expm1, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(ceil_out, Numerics<scalar_t>::ceil, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(trunc_out, Numerics<scalar_t>::trunc, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(round_out, Numerics<scalar_t>::round, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log_out, Numerics<scalar_t>::log, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log10_out, Numerics<scalar_t>::log10, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log1p_out, Numerics<scalar_t>::log1p, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(log2_out, Numerics<scalar_t>::log2, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(rsqrt_out, Numerics<scalar_t>::rsqrt, Real);
IPEX_OUT_FLOAT_UNARY_FUNC_OPS(sqrt_out, Numerics<scalar_t>::sqrt, Real);

IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(tan, Numerics<scalar_t>::tan, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(tanh, Numerics<scalar_t>::tanh, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(atan, Numerics<scalar_t>::atan, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(erf, Numerics<scalar_t>::erf, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(erfc, Numerics<scalar_t>::erfc, Real);
IPEX_OUT_INPLACE_FLOAT_UNARY_FUNC_OPS(exp, Numerics<scalar_t>::exp, Real);

IPEX_ALL_CALLABLE_1_UNARY_OPS(clamp_max_, TensorMinValueOp);
IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(clamp_max_out, TensorMinValueOp);
IPEX_ALL_CALLABLE_1_UNARY_OPS(clamp_min_, TensorMaxValueOp);
IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(clamp_min_out, TensorMaxValueOp);
IPEX_OUT_ALL_CALLABLE_2_UNARY_OPS(clamp_min_max, TensorClampOp);

Tensor & clamp_out(Tensor & result, const Tensor & self,
    optional<Scalar> min, optional<Scalar> max) {
  if (min && max) {
    at::AtenIpexTypeDPCPP::clamp_min_max(result, self, *min, *max);
  } else if (max) {
    at::AtenIpexTypeDPCPP::clamp_max_out(result, self, *max);
  } else if (min) {
    at::AtenIpexTypeDPCPP::clamp_min_out(result, self, *min);
  } else {
    TORCH_CHECK(false, "At least one of 'min' or 'max' must not be None");
  }
  return result;
}

Tensor & clamp_(Tensor & self, optional<Scalar> min, optional<Scalar> max) {
  return at::AtenIpexTypeDPCPP::clamp_out(self, self, min, max);
}

Tensor bitwise_not(const Tensor & self){
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::bitwise_not_out(result, self);
}

Tensor & bitwise_not_(Tensor & self){
  return at::AtenIpexTypeDPCPP::bitwise_not_out(self, self);
}

Tensor & bitwise_not_out(Tensor & out, const Tensor & self){
  auto iter = TensorIterator::unary_op(out, self,
    /*check_mem_overlap=*/true);
  impl::bitwise_not_kernel_sycl(iter);
  #ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(out, self);
  #endif
  return out;
}

Tensor logical_not(const Tensor& self) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::AtenIpexTypeDPCPP::logical_not_out(result, self);
}

Tensor& logical_not_(Tensor& self) {
  return at::AtenIpexTypeDPCPP::logical_not_out(self, self);
}

Tensor& logical_not_out(Tensor& result, const Tensor& self) {
  TensorIterator iter;
  iter.dont_compute_common_dtype();
  iter.set_check_mem_overlap(true);
  iter.add_output(result);
  iter.add_input(self);
  iter.build();
  impl::logical_not_kernel_sycl(iter);
  return result;
}

IPEX_OUT_INT_CALLABLE_1_UNARY_OPS(__and___out, TensorBitAndConstantOp);

Tensor __and__(const Tensor & self, Scalar other) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::__and___out(result, self, other);
}

Tensor & __iand__(Tensor & self, Scalar other) {
  return at::AtenIpexTypeDPCPP::__and___out(self, self, other);
}

IPEX_OUT_INT_CALLABLE_1_UNARY_OPS(__or___out, TensorBitOrConstantOp);

Tensor __or__(const Tensor & self, Scalar other) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::__or___out(result, self, other);
}

Tensor & __ior__(Tensor & self, Scalar other) {
  return at::AtenIpexTypeDPCPP::__or___out(self, self, other);
}

IPEX_OUT_FLOAT_AND_HALF_CALLABLE_0_UNARY_OPS(erfinv_out, TensorErfinvOp);

Tensor & erfinv_(Tensor & self) {
  return at::AtenIpexTypeDPCPP::erfinv_out(self, self);
}

Tensor erfinv(const Tensor & self) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::erfinv_out(result, self);
}

IPEX_OUT_FLOAT_AND_HALF_CALLABLE_0_UNARY_OPS(digamma_out, TensorDigammaOp);

Tensor & digamma_(Tensor & self) {
  return at::AtenIpexTypeDPCPP::digamma_out(self, self);
}

Tensor digamma(const Tensor & self) {
  auto result = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::digamma_out(result, self);
}

IPEX_OUT_ALL_CALLABLE_1_UNARY_OPS(remainder_out, TensorRemainderOp)

Tensor remainder(const Tensor & self, Scalar other) {
  auto out = at::empty_like(self);
  return at::AtenIpexTypeDPCPP::remainder_out(out, self, other);
}

Tensor & remainder(Tensor & self, Scalar other) {
  return at::AtenIpexTypeDPCPP::remainder_out(self, self, other);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
