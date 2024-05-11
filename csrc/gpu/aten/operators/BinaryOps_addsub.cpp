#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/OpMathType.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <oneDNN/oneDNN.h>
#include <utils/DPCPP.h>
#include <utils/Macros.h>
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "EltwiseNaiveKer.h"
#include "Loops.h"
#include "LoopsTemplates.h"

#include <utils/ComputeEngine.h>

using namespace xpu::dpcpp;

namespace at {
namespace impl {

template <typename scalar_t, typename accscalar_t>
struct AddKernelDpcppFunctor {
  scalar_t operator()(accscalar_t a, accscalar_t b) const {
    return a + alpha * b;
  }

  AddKernelDpcppFunctor<scalar_t, accscalar_t>(accscalar_t alpha)
      : alpha(alpha) {}

 private:
  accscalar_t alpha;
};

void add_kernel_dpcpp(TensorIteratorBase& iter, Scalar alpha_scalar) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      at::ScalarType::ComplexHalf,
      iter.dtype(),
      "add",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto alpha = alpha_scalar.to<opmath_t>();
        AddKernelDpcppFunctor<scalar_t, opmath_t> kfn(alpha);
        opmath_gpu_kernel_with_scalars<
            scalar_t,
            scalar_t,
            scalar_t,
            decltype(kfn),
            true>(iter, kfn);
      });
}

void sub_kernel_dpcpp(TensorIteratorBase& iter, Scalar alpha_scalar) {
  return add_kernel_dpcpp(iter, -alpha_scalar);
}

// alpha_check
inline void alpha_check(const TensorIterator& iter, Scalar alpha) {
  TORCH_CHECK(
      !alpha.isBoolean() || iter.dtype() == ScalarType::Bool,
      "Boolean alpha only supported for Boolean results.");
  TORCH_CHECK(
      isFloatingType(iter.dtype()) || isComplexType(iter.dtype()) ||
          alpha.isIntegral(true),
      "For integral input tensors, argument alpha must not be a floating "
      "point number.");
}

// Basic checking for all sub functions.
inline void sub_check(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      self.scalar_type() != kBool || other.scalar_type() != kBool,
      "Subtraction, the `-` operator, with two bool tensors is not supported. "
      "Use the `^` or `logical_xor()` operator instead.");
  TORCH_CHECK(
      self.scalar_type() != kBool && other.scalar_type() != kBool,
      "Subtraction, the `-` operator, with a bool tensor is not supported. "
      "If you are trying to invert a mask, use the `~` or `logical_not()` "
      "operator instead.");
}

} // namespace impl

namespace AtenIpexTypeXPU {

Tensor& add_out(
    const Tensor& _self,
    const Tensor& _other,
    const Scalar& alpha,
    Tensor& result) {
  return binary_out_template<dnnl::algorithm::binary_add>(
      TensorIterator::binary_op,
      result,
      _self,
      _other,
      [=](TensorIteratorBase& iter) {
        impl::alpha_check(iter, alpha);
        impl::add_kernel_dpcpp(iter, alpha);
      },
      ((!alpha.isComplex()) && (1.0 == alpha.to<float>())));
}

Tensor add(const Tensor& _self, const Tensor& _other, const Scalar& alpha) {
  Tensor result;
  return binary_out_template<dnnl::algorithm::binary_add>(
      TensorIterator::binary_op,
      result,
      _self,
      _other,
      [=](TensorIteratorBase& iter) {
        impl::alpha_check(iter, alpha);
        impl::add_kernel_dpcpp(iter, alpha);
      },
      ((!alpha.isComplex()) && (1.0 == alpha.to<float>())));
}

Tensor& add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return at::AtenIpexTypeXPU::add_out(self, other, alpha, self);
}

Tensor add(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return at::AtenIpexTypeXPU::add(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& add_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return at::AtenIpexTypeXPU::add_(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& sub_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& result) {
  impl::sub_check(self, other);
  return binary_out_template<dnnl::algorithm::binary_sub>(
      TensorIterator::binary_op,
      result,
      self,
      other,
      [=](TensorIteratorBase& iter) {
        impl::alpha_check(iter, alpha);
        impl::sub_kernel_dpcpp(iter, alpha);
      },
      ((!alpha.isComplex()) && (1.0 == alpha.to<float>())));
}

Tensor rsub(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  Tensor out;
  auto iter = TensorIterator::binary_op(out, other, self);
  out = iter.output();
  return AtenIpexTypeXPU::sub_out(other, self, alpha, out);
}

} // namespace AtenIpexTypeXPU

} // namespace at
