#include <ATen/Config.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/TypeProperties.h>
#include "CatImpl.h"

namespace at {
namespace AtenIpexTypeXPU {

Tensor& cat_out(const ITensorListRef& container, int64_t dim, Tensor& out) {
  cat_(container, dim, out);
  return out;
}

Tensor cat(const ITensorListRef& tensors, int64_t dim) {
  auto high_type = at::native::result_type(tensors);
  auto out = at::empty({0}, tensors.front().options().dtype(high_type));
  return at::AtenIpexTypeXPU::cat_out(tensors, dim, out);
}

} // namespace AtenIpexTypeXPU
} // namespace at
