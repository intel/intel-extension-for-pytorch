#include "torch_ipex/csrc/cpu/SparseOPs.h"
#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/ipex_sparse_tensor_impl.h"

namespace torch_ipex {
namespace cpu {

//#define DBG
#if defined(DBG)
#define DEBUG(fmt) printf(fmt);
#else
#define DEBUG(fmt)
#endif

at::Tensor AtenIpexCPUSparse::_indices(const at::Tensor & self) {
  DEBUG("AtenIpexCPUSparse::_indices\n");
  TORCH_INTERNAL_ASSERT(self.layout() == c10::kSparse);
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->indices();
}

at::Tensor AtenIpexCPUSparse::_values(const at::Tensor & self) {
  DEBUG("AtenIpexCPUSparse::_values\n");
  TORCH_INTERNAL_ASSERT(self.layout() == c10::kSparse);
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->values();
}

int64_t AtenIpexCPUSparse::sparse_dim(const at::Tensor & self) {
  DEBUG("AtenIpexCPUSparse::sparse_dim\n");
  TORCH_INTERNAL_ASSERT(self.layout() == c10::kSparse);
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->sparse_dim();
}

int64_t AtenIpexCPUSparse::dense_dim(const at::Tensor & self) {
  DEBUG("AtenIpexCPUSparse::dense_dim\n");
  TORCH_INTERNAL_ASSERT(self.layout() == c10::kSparse);
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->dense_dim();
}

int64_t AtenIpexCPUSparse::_nnz(const at::Tensor & self) {
  DEBUG("AtenIpexCPUSparse::_nnz\n");
  TORCH_INTERNAL_ASSERT(self.layout() == c10::kSparse);
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->nnz();
}

bool AtenIpexCPUSparse::is_coalesced(const at::Tensor & self) {
  DEBUG("AtenIpexCPUSparse::is_coalesced\n");
  TORCH_INTERNAL_ASSERT(self.layout() == c10::kSparse);
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->coalesced();
}

at::Tensor & AtenIpexCPUSparse::_coalesced_(at::Tensor & self, bool coalesced) {
  DEBUG("AtenIpexCPUSparse::_coalesced_\n");
  TORCH_INTERNAL_ASSERT(self.layout() == c10::kSparse);
  IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->set_coalesced(coalesced);
  return self;
}

// Align with at::clone_sparse
at::Tensor AtenIpexCPUSparse::clone(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
  DEBUG("AtenIpexCPUSparse::clone\n");
  TORCH_INTERNAL_ASSERT(self.layout() == c10::kSparse);
  TORCH_INTERNAL_ASSERT(!memory_format.has_value());
  TORCH_INTERNAL_ASSERT(at::impl::variable_is_excluded());

  // Create and resize sparse tensor
  auto _tensor = at::detail::make_tensor<IPEXSparseTensorImpl>(
      at::TensorTypeSet(at::TensorTypeId::SparseDPCPPTensorId), self.dtype());
  IPEXSparseTensorImpl::get_ipex_sparse_impl(_tensor)->resize_and_clear_(self.sparse_dim(), self.dense_dim(), self.sizes());
  // Copy indices and values
  at::sparse::copy_into_sparse(_tensor, self._indices(), self._values(), true);
  return _tensor._coalesced_(self.is_coalesced());
}

// Align with at::new_with_dims_and_tensor_sparse
at::Tensor AtenIpexCPUSparse::_sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim,
    at::IntArrayRef size, const at::Tensor & indices, const at::Tensor & values, const at::TensorOptions & options) {
  TORCH_INTERNAL_ASSERT(indices.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT(values.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT(options.device().type() == at::DeviceType::DPCPP);
  //TODO
  auto _tensor = at::detail::make_tensor<IPEXSparseTensorImpl>(
      at::TensorTypeSet(at::TensorTypeId::SparseDPCPPTensorId), options.dtype());
  IPEXSparseTensorImpl::get_ipex_sparse_impl(_tensor)->resize_(sparse_dim, dense_dim, size);
  // NOTE: There is no guarantee that `indices` and `values` don't contain AutogradMeta. However,
  // we want to maintain the invariant that `indices_` and `values_` of a sparse tensor don't
  // contain AutogradMeta, and to achieve that we shallow-copy `indices` and `values` here.
  auto indices_shallow_copy = at::sparse::LongTensor(indices.unsafeGetTensorImpl()->shallow_copy_and_detach(
    /*version_counter=*/indices.unsafeGetTensorImpl()->version_counter(),
    /*allow_tensor_metadata_change=*/true));
  auto values_shallow_copy = at::Tensor(values.unsafeGetTensorImpl()->shallow_copy_and_detach(
    /*version_counter=*/values.unsafeGetTensorImpl()->version_counter(),
    /*allow_tensor_metadata_change=*/true));
  at::sparse::alias_into_sparse(_tensor, indices_shallow_copy, values_shallow_copy);
  return _tensor;
}

}  // namespace cpu
}  // namespace torch_ipex
