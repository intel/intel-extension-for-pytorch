#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/custom_class.h>

#include <core/DPCPPUtils.h>
#include <utils/ParamUtils.h>
#include <oneDNN/oneDNN.h>


using namespace xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeQuantizedXPU {

Tensor qAdd(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
  auto a = at::dequantize(qa);
  auto b = at::dequantize(qb);
  auto c = at::add(a, b, 1.f);

  auto qc = at::quantize_per_tensor(c, scale, zero_point, qa.scalar_type());
  return qc;
}

Tensor qAddRelu(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
  auto a = at::dequantize(qa);
  auto b = at::dequantize(qb);
  auto c = at::add(a, b, 1.f);

  // use oneDNN reorder of s8 to u8 to do relu's thing!
  auto qc = at::quantize_per_tensor(c, scale, zero_point, ScalarType::QUInt8);

  return qc;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedXPU, m) {
  m.impl("add", qAdd);
  m.impl("add_relu", qAddRelu);
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
