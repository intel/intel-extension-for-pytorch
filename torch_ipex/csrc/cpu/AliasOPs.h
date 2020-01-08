#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {
namespace cpu {

class AtenIpexCPUAlias {
 public:
  // aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)
  static at::Tensor as_strided(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset);
  // aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]
  static std::vector<at::Tensor> chunk(const at::Tensor & self, int64_t chunks, int64_t dim);
  // aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
  static at::Tensor diagonal(const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2);
  // aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)
  static at::Tensor expand(const at::Tensor & self, at::IntArrayRef size, bool implicit);
  // aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)
  static at::Tensor narrow(const at::Tensor & self, int64_t dim, int64_t start, int64_t length);
  // aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
  static at::Tensor permute(const at::Tensor & self, at::IntArrayRef dims);
  // aten::numpy_T(Tensor(a) self) -> Tensor(a)
  static at::Tensor numpy_T(const at::Tensor & self);
  // aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
  static at::Tensor select(const at::Tensor & self, int64_t dim, int64_t index);
  // aten::slice.Tensor(Tensor(a) self, int dim=0, int start=0, int end=9223372036854775807, int step=1) -> Tensor(a)
  static at::Tensor slice(const at::Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step);
  // aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]
  static std::vector<at::Tensor> split(const at::Tensor & self, int64_t split_size, int64_t dim);
  // aten::squeeze(Tensor(a) self) -> Tensor(a)
  static at::Tensor squeeze(const at::Tensor & self);
  // aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
  static at::Tensor squeeze(const at::Tensor & self, int64_t dim);
  // aten::t(Tensor(a) self) -> Tensor(a)
  static at::Tensor t(const at::Tensor & self);
  // aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
  static at::Tensor transpose(const at::Tensor & self, int64_t dim0, int64_t dim1);
  // aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
  static at::Tensor unsqueeze(const at::Tensor & self, int64_t dim);
  // aten::_indices(Tensor(a) self) -> Tensor(a)
  static at::Tensor _indices(const at::Tensor & self);
  // aten::_values(Tensor(a) self) -> Tensor(a)
  static at::Tensor _values(const at::Tensor & self);
  // aten::indices(Tensor(a) self) -> Tensor(a)
  static at::Tensor indices(const at::Tensor & self);
  // aten::values(Tensor(a) self) -> Tensor(a)
  static at::Tensor values(const at::Tensor & self);
  // aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]
  static std::vector<at::Tensor> unbind(const at::Tensor & self, int64_t dim);
  // aten::view(Tensor(a) self, int[] size) -> Tensor(a)
  static at::Tensor view(const at::Tensor & self, at::IntArrayRef size);
  // aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)
  static at::Tensor unfold(const at::Tensor & self, int64_t dimension, int64_t size, int64_t step);
  // aten::alias(Tensor(a) self) -> Tensor(a)
  static at::Tensor alias(const at::Tensor & self);

};

}  // namespace cpu
}  // namespace torch_ipex

