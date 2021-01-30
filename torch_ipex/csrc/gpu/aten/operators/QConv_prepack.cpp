#include <ATen/core/op_registration/op_registration.h>
#include <core/DPCPPUtils.h>
#include <core/Runtime.h>

#include <utils/ParamUtils.h>

#include "QUtil.h"

using namespace dnnl;
using namespace at::dpcpp;
using namespace at::native;

c10::intrusive_ptr<ConvPackedParamsBase<2>> at::AtenIpexTypeQuantizedXPU::PackedConvWeightQDPCPP::prepack(
        at::Tensor weight,
        c10::optional<at::Tensor> bias,
        torch::List<int64_t> stride,
        torch::List<int64_t> padding,
        torch::List<int64_t> dilation,
        int64_t groups) {
  c10::intrusive_ptr<ConvPackedParamsBase<2>> ret_ptr = c10::make_intrusive<PackedConvWeightQDPCPP>(
      at::AtenIpexTypeQuantizedXPU::PackedConvWeightQDPCPP{weight, bias, stride, padding, dilation, groups});
  return ret_ptr;
}

namespace at {
namespace AtenIpexTypeQuantizedXPU {

c10::intrusive_ptr<ConvPackedParamsBase<2>> dpcppConvPrepack(
    Tensor weight,
    c10::optional<Tensor> bias,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups) {
  // This is just align with Pytorch Python API!
  auto ret_ptr = PackedConvWeightQDPCPP::prepack(
          weight, bias, stride, padding, dilation, groups);
  return ret_ptr;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedXPU, m) {
  m.impl("conv2d_prepack", TORCH_FN(dpcppConvPrepack));
}

TORCH_LIBRARY_IMPL(quantized, XPU, m) {
  m.impl("conv2d_prepack", TORCH_FN(dpcppConvPrepack));
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
