#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {
namespace cpu {

class AtenIpexCPUSparse {
 public:
  // aten::_indices(Tensor(a) self) -> Tensor(a)
  static at::Tensor _indices(const at::Tensor & self);
  // aten::_values(Tensor(a) self) -> Tensor(a)
  static at::Tensor _values(const at::Tensor & self);
  static int64_t sparse_dim(const at::Tensor & self);
  static int64_t dense_dim(const at::Tensor & self);
  static int64_t _nnz(const at::Tensor & self);
  static bool is_coalesced(const at::Tensor & self);
  static at::Tensor & _coalesced_(at::Tensor & self, bool coalesced);
  static at::Tensor clone(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format);
};

}  // namespace cpu
}  // namespace torch_ipex
