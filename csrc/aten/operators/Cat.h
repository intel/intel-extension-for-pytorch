#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <oneDNN/oneDNN.h>

#include <runtime/DPCPP.h>

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

void dnnl_cat(Tensor& output, TensorList inputs, int numInputs, int dimension);

} // namespace impl
} // namespace AtenIpexTypeXPU
} // namespace at
