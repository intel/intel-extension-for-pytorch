#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/AtenIpexTypeXPU.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>

#include <utils/ParamUtils.h>

#include "InnerProduct.h"
#include "QUtil.h"

using namespace dnnl;
using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeQuantizedXPU {

using namespace impl;

at::Tensor dpcppLinear(
    Tensor input,
    const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto pack_ptr =
      dynamic_cast<PackedLinearWeightQDPCPP*>(packed_weight.get());
  at::Tensor weight = pack_ptr->weight;
  at::Tensor bias;
  if (pack_ptr->bias_.has_value()) {
    bias = pack_ptr->bias_.value();
  } else {
    bias = at::empty({0}, input.options());
  }

Tensor output;
  if(weight.is_quantized()){
    output = at::addmm(
        bias,
        input,
        weight,
        bias.is_quantized() ? bias.q_scale() : 1.f,
        output_scale);
  } else {
    output = at::empty({input.size(0), weight.size(0)}, weight.options());
  
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    bool use_bias = pack_ptr->bias_.has_value();
    inner_product(
        M,
        N,
        K,
        output,
        input,
        weight,
        bias,
        use_bias);
  }
  return output;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedXPU, m) {
  m.impl("quantized::linear", dpcppLinear);
}

} //namespace AtenIpexTypeQuantizedXPU 
} // namespace at
