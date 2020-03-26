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

at::Tensor AtenIpexCPUSparse::clone(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
  TORCH_INTERNAL_ASSERT(self.layout() == c10::kSparse);
  TORCH_INTERNAL_ASSERT(!memory_format.has_value());
  TORCH_INTERNAL_ASSERT(at::impl::variable_is_excluded());

  auto&& cpu_self = bridge::shallowFallbackToCPUTensor(self);
  auto&& cpu_result = at::clone(cpu_self, memory_format);
  return bridge::shallowUpgradeToDPCPPTensor(cpu_result);
}

at::Tensor AtenIpexCPUSparse::_sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim,
    at::IntArrayRef size, const at::Tensor & indices, const at::Tensor & values, const at::TensorOptions & options) {
  TORCH_INTERNAL_ASSERT(indices.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT(values.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT(options.device().type() == at::DeviceType::DPCPP);

  auto&& cpu_indices = bridge::shallowFallbackToCPUTensor(indices);
  auto&& cpu_values = bridge::shallowFallbackToCPUTensor(values);
  at::TensorOptions cpu_options = options.device(at::DeviceType::CPU);
  auto&& cpu_result = at::_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, cpu_indices, cpu_values, cpu_options);
  return bridge::shallowUpgradeToDPCPPTensor(cpu_result);
}

at::Tensor & AtenIpexCPUSparse::add_(at::Tensor & self, const at::Tensor & other, at::Scalar alpha) {
  TORCH_INTERNAL_ASSERT(other.layout() == c10::kSparse);
  TORCH_INTERNAL_ASSERT(self.layout() == c10::kStrided || self.layout() ==c10::kSparse);

  auto&& cpu_self = bridge::shallowFallbackToCPUTensor(self);
  auto&& cpu_other = bridge::shallowFallbackToCPUTensor(other);
  auto&& cpu_result = cpu_self.add_(cpu_other, alpha);
  bridge::shallowUpgradeToDPCPPTensorAW(self, cpu_self);
  return self;
}

at::Tensor AtenIpexCPUSparse::empty(at::IntArrayRef size, const at::TensorOptions & options, c10::optional<at::MemoryFormat> memory_format) {
  TORCH_INTERNAL_ASSERT(options.device().type() == at::DeviceType::DPCPP);
  at::TensorOptions cpu_options = options.device(at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(memory_format.value_or(c10::MemoryFormat::Contiguous) == c10::MemoryFormat::Contiguous);
  auto&& cpu_result = at::empty(size, cpu_options, memory_format);
  return bridge::shallowUpgradeToDPCPPTensor(cpu_result);
}

at::Tensor & AtenIpexCPUSparse::copy_sparse_to_sparse_(at::Tensor & self, const at::Tensor & src, bool non_blocking) {
  TORCH_INTERNAL_ASSERT(self.layout() == c10::kSparse);
  TORCH_INTERNAL_ASSERT(src.layout() == c10::kSparse);
  auto&& cpu_self = bridge::shallowFallbackToCPUTensor(self);
  auto&& cpu_src = bridge::shallowFallbackToCPUTensor(src);
  // NOTICE:
  // This is one significant inplace operation, when self and src are not same tensor, it will
  // generate new indices and values to output. Then output and original input become irrelevant.
  //
  // Specially, when self is one empty sparse tensor, it will bind new indices and values
  // to output, while original self still keeps empty. We have added "special path" in 
  // `shallowUpgradeToDPCPPTensorAW` to handle this issue.
  //
  auto&& cpu_result = at::copy_sparse_to_sparse_(cpu_self, cpu_src, non_blocking);
  bridge::shallowUpgradeToDPCPPTensorAW(self, cpu_self);
  return self;
}

}  // namespace cpu
}  // namespace torch_ipex
