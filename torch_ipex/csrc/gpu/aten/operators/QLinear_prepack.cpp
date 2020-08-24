#include <ATen/core/op_registration/op_registration.h>
#include <core/DPCPPUtils.h>
#include <core/Runtime.h>

#include <utils/ParamUtils.h>

#include "QUtil.h"

using namespace dnnl;
using namespace at::dpcpp;
using namespace at::native;

namespace caffe2 {

CAFFE_KNOWN_TYPE(at::AtenIpexTypeDPCPP::PackedLinearWeightQDPCPP);
}

namespace at {
namespace AtenIpexTypeDPCPP {

at::Tensor dpcppLinearPrepack(Tensor weight, c10::optional<Tensor> bias) {
  // This is just align with FBGEMM INT8 and Pytorch Python API!
  auto ret_ptr = std::make_unique<PackedLinearWeightQDPCPP>(
      PackedLinearWeightQDPCPP{weight, bias});
  return at::cpp_custom_type_hack::create(std::move(ret_ptr), weight.options());
}

static auto registry = c10::RegisterOperators().op(
    "quantized::linear_prepack(Tensor W, Tensor? B=None) -> Tensor W_prepack",
    c10::RegisterOperators::options()
        .kernel<decltype(dpcppLinearPrepack), &dpcppLinearPrepack>(
            DispatchKey::QuantizedDPCPPTensorId));

} // namespace AtenIpexTypeDPCPP
} // namespace at
