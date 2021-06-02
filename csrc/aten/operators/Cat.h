#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <oneDNN/oneDNN.h>
#ifdef USE_PRIMITIVE_CACHE
#include <oneDNN/LRUCache.h>
#endif

#include <core/DPCPP.h>

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void dnnl_cat(Tensor& output, TensorList inputs, int numInputs, int dimension);

} // namespace impl
} // namespace AtenIpexTypeXPU
} // namespace at
