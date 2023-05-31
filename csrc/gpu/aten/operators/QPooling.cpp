#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>

#include "comm/ParamUtils.h"
#include "comm/RegistrationDeclarations.h"

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
  auto output = at::_empty_affine_quantized(
      {0},
      qx.options(),
      qx.q_scale(),
      qx.q_zero_point(),
      MemoryFormat::Contiguous); // Relu fusion?
  auto indices = at::empty({0}, qx.options().dtype(kLong));
  auto output_and_indices = at::AtenIpexTypeXPU::max_pool2d_with_indices_out(
      qx, kernel_size, stride, padding, dilation, ceil_mode, output, indices);

  return std::get<0>(output_and_indices);
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
