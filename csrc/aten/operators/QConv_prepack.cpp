#include <ATen/core/op_registration/op_registration.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>

#include "comm/ParamUtils.h"

#include <quantized/QUtil.h>

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;

template <int kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> at::
    AtenIpexTypeQuantizedXPU::PackedConvWeightQDPCPP<kSpatialDim>::prepack(
        at::Tensor weight,
        c10::optional<at::Tensor> bias,
        torch::List<int64_t> stride,
        torch::List<int64_t> padding,
        torch::List<int64_t> output_padding,
        torch::List<int64_t> dilation,
        int64_t groups,
        bool transpose) {
  c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> ret_ptr =
      c10::make_intrusive<PackedConvWeightQDPCPP<kSpatialDim>>(
          at::AtenIpexTypeQuantizedXPU::PackedConvWeightQDPCPP<kSpatialDim>{
              weight,
              bias,
              stride,
              padding,
              output_padding,
              dilation,
              groups,
              transpose});
  return ret_ptr;
}
template struct at::AtenIpexTypeQuantizedXPU::PackedConvWeightQDPCPP<2>;
template struct at::AtenIpexTypeQuantizedXPU::PackedConvWeightQDPCPP<3>;
namespace at {
namespace AtenIpexTypeQuantizedXPU {

c10::intrusive_ptr<ConvPackedParamsBase<2>> dpcppConvPrepack2d(
    Tensor weight,
    c10::optional<Tensor> bias,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups) {
  // This is just align with Pytorch Python API!
  torch::List<int64_t> output_padding;
  int dim = 2;
  output_padding.reserve(dim);
  for (int idx = 0; idx < dim; ++idx) {
    output_padding.push_back((int64_t)0);
  }
  auto ret_ptr = PackedConvWeightQDPCPP<2>::prepack(
      weight,
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      /*transpose=*/false);
  return ret_ptr;
}

c10::intrusive_ptr<ConvPackedParamsBase<3>> dpcppConvPrepack3d(
    Tensor weight,
    c10::optional<Tensor> bias,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups) {
  torch::List<int64_t> output_padding;
  int dim = 3;
  output_padding.reserve(dim);
  for (int idx = 0; idx < dim; ++idx) {
    output_padding.push_back((int64_t)0);
  }
  auto ret_ptr = PackedConvWeightQDPCPP<3>::prepack(
      weight,
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      /*transpose=*/false);
  return ret_ptr;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedXPU, m) {
  m.impl("conv2d_prepack", TORCH_FN(dpcppConvPrepack2d));
  m.impl("conv3d_prepack", TORCH_FN(dpcppConvPrepack3d));
}

TORCH_LIBRARY_IMPL(quantized, XPU, m) {
  m.impl("conv2d_prepack", TORCH_FN(dpcppConvPrepack2d));
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
