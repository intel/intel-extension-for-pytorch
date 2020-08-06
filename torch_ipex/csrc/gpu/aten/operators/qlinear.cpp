#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <ATen/core/op_registration/op_registration.h>
#include <core/DPCPPUtils.h>
#include <core/Runtime.h>

#include <utils/ParamUtils.h>
#include <ATen/aten_ipex_type_dpcpp.h>

#include "qutil.h"

using namespace dnnl;
using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {

using namespace impl;

at::Tensor dpcppLinear(
    Tensor input,
    Tensor packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto& pack_ptr =
      cpp_custom_type_hack::cast<PackedLinearWeightQDPCPP>(packed_weight);
  at::Tensor bias;
  if (pack_ptr.bias.has_value())
    bias = pack_ptr.bias.value();

  auto output = at::addmm(
      bias,
      input,
      packed_weight,
      bias.is_quantized() ? bias.q_scale() : 1.f,
      output_scale);

  return output;
}

static auto registry = c10::RegisterOperators().op(
    "quantized::linear(Tensor X, Tensor W_prepack, float Y_scale_i, int "
    "Y_zero_point_i) -> Tensor Y",
    c10::RegisterOperators::options()
        .kernel<decltype(dpcppLinear), &dpcppLinear>(
            DispatchKey::QuantizedDPCPPTensorId));

} // namespace AtenIpexTypeDPCPP
} // namespace at
