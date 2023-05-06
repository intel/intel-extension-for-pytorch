#include <ATen/ATen.h>

namespace at {
namespace AtenIpexTypeSparseXPU {
Tensor _to_dense(const Tensor& self, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(
      !dtype.has_value(), "dtype argument is not supported by sparse_to_dense");
  Tensor dst = at::zeros(self.sizes(), self.options().layout(kStrided));
  return dst.add_(self);
}

} // namespace AtenIpexTypeSparseXPU
} // namespace at