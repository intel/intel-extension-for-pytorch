#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>
#include <core/Memory.h>
#include <core/detail/ListUtils.h>
#include <runtime/Utils.h>

inline void alpha_check(const ScalarType dtype, const Scalar& alpha) {
  TORCH_CHECK(
      !alpha.isBoolean() || dtype == ScalarType::Bool,
      "Boolean alpha only supported for Boolean results.");
  TORCH_CHECK(
      isFloatingType(dtype) || isComplexType(dtype) || alpha.isIntegral(true),
      "For integral input tensors, argument alpha must not be a floating point number.");
  TORCH_CHECK(
      isComplexType(dtype) || !alpha.isComplex(),
      "For non-complex input tensors, argument alpha must not be a complex number.")
}

namespace at {
namespace AtenIpexTypeSparseXPU {

Tensor add(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // TODO: Why?! Can't we just flip the order here...
  TORCH_CHECK(
      !(self.is_sparse() && !other.is_sparse()),
      "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  auto commonDtype = at::result_type(self, other);
  alpha_check(commonDtype, alpha);
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  return at::add_out(result, self, other, alpha); // redispatch!
}

Tensor& add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return at::add_out(self, self, other, alpha); // redispatch!
}

} // namespace AtenIpexTypeSparseXPU
} // namespace at
