#include <ATen/ATen.h>

namespace at {
namespace AtenIpexTypeSparseXPU {

Tensor flatten_indices(
    const Tensor& indices,
    IntArrayRef full_size,
    bool force_clone /*= false*/);

}
} // namespace at
