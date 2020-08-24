#include <ATen/core/op_registration/op_registration.h>
#include <core/DPCPPUtils.h>
#include <core/Runtime.h>

#include <utils/ParamUtils.h>

#include "QUtil.h"

using namespace dnnl;
using namespace at::dpcpp;
using namespace at::native;

namespace caffe2 {

CAFFE_KNOWN_TYPE(at::AtenIpexTypeDPCPP::PackedConvWeightQDPCPP);
}

namespace at {
namespace AtenIpexTypeDPCPP {

at::Tensor dpcppConvPrepack(
    Tensor weight,
    c10::optional<Tensor> bias,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups) {
  // This is just align with FBGEMM INT8 and Pytorch Python API!
  auto ret_ptr = std::make_unique<PackedConvWeightQDPCPP>(
      PackedConvWeightQDPCPP{weight, bias});
  return at::cpp_custom_type_hack::create(std::move(ret_ptr), weight.options());
}

static auto registry = c10::RegisterOperators().op(
    "quantized::conv2d_prepack",
    c10::RegisterOperators::options()
        .kernel<decltype(dpcppConvPrepack), &dpcppConvPrepack>(
            DispatchKey::QuantizedDPCPPTensorId));

} // namespace AtenIpexTypeDPCPP
} // namespace at
