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

#if 0
SparseTensor& add_out_sparse_cuda(const SparseTensor& t, const SparseTensor& src, const Scalar& value, SparseTensor& r_) {
  if (!t.is_sparse()) {
    return add_out_dense_sparse_cuda(r_, t, src, value);
  }

  // TODO: This test seems a bit goofy
  TORCH_CHECK(src.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");

  TORCH_CHECK(t.is_cuda(), "add: expected 'self' to be CUDA, but got CPU");
  TORCH_CHECK(src.is_cuda(), "add: expected 'other' to be CUDA, but got CPU");
  TORCH_CHECK(r_.is_cuda(), "add: expected 'out' to be CUDA, but got CPU");

  TORCH_CHECK(cuda::check_device({r_, t, src}));

  auto commonDtype = at::result_type(t, src);
  TORCH_CHECK(canCast(commonDtype, r_.scalar_type()), "Can't convert result type ", commonDtype, " to output ", r_.scalar_type());

  TORCH_CHECK(t.sizes().equals(src.sizes()), "add: expected 'self' and 'other' to have same size, but ", t.sizes(), " != ", src.sizes());

  if (src._nnz() == 0) {
    return copy_sparse_to_sparse_(r_, t);
  }
  if (t._nnz() == 0) {
    return mul_out_sparse_scalar(r_, src, value);
  }

  TORCH_CHECK(is_same_density(t, src), "add: expected 'self' and 'other' to have same density, but 'self' has ", t.sparse_dim(), " sparse dimensions while 'other' has ", src.sparse_dim(), " sparse dimensions");

  // We deliberately choose to simply concat the indices and values tensors
  // rather than merging them. This removes the need to synchronously fetch nnz
  // at the end of the operation, at the cost of having a non-coalesced result.
  // This trade-off is preferable for the common use-case of gradient accumulation.
  Tensor t_indices_ = t._indices();
  Tensor s_indices_ = src._indices();

  Tensor t_values_ = t._values().to(commonDtype);
  Tensor s_values_ = src._values().to(commonDtype);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16, commonDtype, "add_out_sparse_cuda", [&] {
      if (value.to<scalar_t>() != scalar_t(1)) {
        s_values_ = s_values_.mul(value);
      }
  });
  Tensor r_indices_ = at::cat({t_indices_, s_indices_}, 1);
  Tensor r_values_ = at::cat({t_values_, s_values_}, 0);

  if (r_.scalar_type() != commonDtype) {
    SparseTensor promoted = at::empty({0}, r_.options().dtype(commonDtype));
    promoted.resize_as_(src);
    alias_into_sparse(promoted, r_indices_, r_values_);
    // performs the addition under the common dtype.
    promoted = promoted.coalesce();
    r_values_ = promoted._values().to(r_.scalar_type());
    r_indices_ = promoted._indices();
  } else {
    r_.resize_as_(src);
  }

  alias_into_sparse(r_, r_indices_, r_values_);

  // Prevent unbounded growth of nnz
  // TODO: Improved heuristic on when to coalesce or remove need to coalesce
  if (r_._nnz() > r_.numel()) {
    auto c = r_.coalesce();
    alias_into_sparse(r_, c._indices(), c._values());
  }

  return r_;
}
#endif

} // namespace AtenIpexTypeSparseXPU
} // namespace at