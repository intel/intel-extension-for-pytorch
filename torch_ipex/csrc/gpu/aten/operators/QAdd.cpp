#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <ATen/core/op_registration/op_registration.h>
#include <core/DPCPPUtils.h>
#include <core/Runtime.h>

#include <utils/ParamUtils.h>

#include <ATen/aten_ipex_type_dpcpp.h>

using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {

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

  // use mkldnn reorder of s8 to u8 to do relu's thing!
  auto qc = at::quantize_per_tensor(c, scale, zero_point, ScalarType::QUInt8);

  return qc;
}

static auto registry =
    c10::RegisterOperators()
        .op("quantized::add(Tensor qa, Tensor qb, float scale, int "
            "zero_point)-> Tensor qc",
            c10::RegisterOperators::options().kernel<decltype(qAdd), &qAdd>(
                DispatchKey::QuantizedDPCPPTensorId))
        .op("quantized::add_relu(Tensor qa, Tensor qb, float scale, int "
            "zero_point)-> Tensor qc",
            c10::RegisterOperators::options()
                .kernel<decltype(qAddRelu), &qAddRelu>(
                    DispatchKey::QuantizedDPCPPTensorId));

} // namespace AtenIpexTypeDPCPP
} // namespace at
