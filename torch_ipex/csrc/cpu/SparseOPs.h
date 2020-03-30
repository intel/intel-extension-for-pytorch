#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {
namespace cpu {

class AtenIpexCPUSparse {
 public:
  static at::Tensor _indices(const at::Tensor & self);
  static at::Tensor _values(const at::Tensor & self);
  static int64_t sparse_dim(const at::Tensor & self);
  static int64_t dense_dim(const at::Tensor & self);
  static int64_t _nnz(const at::Tensor & self);
  static bool is_coalesced(const at::Tensor & self);
  static at::Tensor & _coalesced_(at::Tensor & self, bool coalesced);
  static at::Tensor clone(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format);
  static at::Tensor _sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim,
      at::IntArrayRef size, const at::Tensor & indices, const at::Tensor & values, const at::TensorOptions & options);
  static at::Tensor & add_(at::Tensor & self, const at::Tensor & other, at::Scalar alpha);
  static at::Tensor empty(at::IntArrayRef size, const at::TensorOptions & options, c10::optional<at::MemoryFormat> memory_format);
  static at::Tensor & copy_sparse_to_sparse_(at::Tensor & self, const at::Tensor & src, bool non_blocking);
  static at::Tensor & zero_(at::Tensor & self);
  static at::Tensor to_dense(const at::Tensor & self);
};

}  // namespace cpu
}  // namespace torch_ipex
