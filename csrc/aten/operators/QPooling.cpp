#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <runtime/DPCPPUtils.h>
#include <oneDNN/oneDNN.h>

#include "comm/ParamUtils.h"


using namespace dnnl;
using namespace xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeQuantizedXPU {

at::Tensor quantized_max_pool2d(
    const Tensor& qx,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  auto output_and_indices = at::max_pool2d_with_indices(
      qx, kernel_size, stride, padding, dilation, ceil_mode);
  auto output = std::get<0>(output_and_indices);

  return output;
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
