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
  auto ipex_sparse_impl = IPEXSparseTensorImpl::get_ipex_sparse_impl(_tensor);
  ipex_sparse_impl->resize_and_clear_(self.sparse_dim(), self.dense_dim(), self.sizes());
  // Copy indices and values
  at::sparse::copy_into_sparse(_tensor, self._indices(), self._values(), true);
  return _tensor._coalesced_(self.is_coalesced());
}

}  // namespace cpu
}  // namespace torch_ipex
