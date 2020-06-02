#include "SparseOPs.h"
#include "ipex_sparse_tensor_impl.h"

namespace torch_ipex {
namespace cpu {

int64_t AtenIpexCPUSparse::sparse_dim(const at::Tensor & self) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.layout() == c10::kSparse);
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->sparse_dim();
}

int64_t AtenIpexCPUSparse::dense_dim(const at::Tensor & self) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.layout() == c10::kSparse);
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->dense_dim();
}

int64_t AtenIpexCPUSparse::_dimI(const at::Tensor & self) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.layout() == c10::kSparse);
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->sparse_dim();
}

int64_t AtenIpexCPUSparse::_dimV(const at::Tensor & self) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.layout() == c10::kSparse);
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->dense_dim();
}

int64_t AtenIpexCPUSparse::_nnz(const at::Tensor & self) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.layout() == c10::kSparse);
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->nnz();
}

bool AtenIpexCPUSparse::is_coalesced(const at::Tensor & self) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.layout() == c10::kSparse);
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->coalesced();
}

at::Tensor & AtenIpexCPUSparse::_coalesced_(at::Tensor & self, bool coalesced) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.layout() == c10::kSparse);
  IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->set_coalesced(coalesced);
  return self;
}

at::Tensor AtenIpexCPUSparse::_indices(const at::Tensor & self) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.layout() == c10::kSparse);
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->indices();
}

at::Tensor AtenIpexCPUSparse::_values(const at::Tensor & self) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.layout() == c10::kSparse);
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->values();
}

at::Tensor AtenIpexCPUSparse::indices(const at::Tensor& self) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.is_coalesced(),
           "Cannot get indices on an uncoalesced tensor, please call .coalesce() first");
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->indices().alias();
}

at::Tensor AtenIpexCPUSparse::values(const at::Tensor& self) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.is_coalesced(),
           "Cannot get values on an uncoalesced tensor, please call .coalesce() first");
  return IPEXSparseTensorImpl::get_ipex_sparse_impl(self)->values().alias();
}

}  // namespace cpu
}  // namespace torch_ipex
