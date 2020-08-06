#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <ATen/core/op_registration/op_registration.h>
#include <core/DPCPPUtils.h>
#include <core/Runtime.h>

#include <utils/ParamUtils.h>

#include "Conv.h"
#include "qutil.h"

using namespace mkldnn;
using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {

using namespace impl;

at::Tensor dpcppConv(
    Tensor input,
    Tensor packed_weight,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point) {
  auto& pack_ptr =
      cpp_custom_type_hack::cast<PackedConvWeightQDPCPP>(packed_weight);

  at::Tensor weight = pack_ptr.weight;
  at::Tensor bias;
  if (pack_ptr.bias.has_value())
    bias = pack_ptr.bias.value();

  conv_attr_t attr(
      static_cast<float>(output_scale),
      static_cast<float>(0.f),
      static_cast<float>(0.f),
      0);

  Tensor output = _empty_affine_quantized(
      conv_output_size(
          input.sizes(),
          weight.sizes(),
          padding.vec(),
          stride.vec(),
          {1, 1},
          groups),
      device(kDPCPP).dtype(kQInt8),
      output_scale,
      output_zero_point,
      MemoryFormat::Contiguous);
  output = convolution(
      output,
      input,
      weight,
      bias,
      padding.vec(),
      stride.vec(),
      {1, 1},
      groups,
      attr);

  return output;
}

at::Tensor dpcppConvRelu(
    Tensor input,
    Tensor packed_weight,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point) {
  auto& pack_ptr =
      cpp_custom_type_hack::cast<PackedConvWeightQDPCPP>(packed_weight);

  at::Tensor weight = pack_ptr.weight;
  at::Tensor bias;
  if (pack_ptr.bias.has_value())
    bias = pack_ptr.bias.value();

  conv_attr_t attr(
      static_cast<float>(output_scale),
      static_cast<float>(0.f),
      static_cast<float>(0.f),
      conv_attr_t::kind_with_relu);

  Tensor output = _empty_affine_quantized(
      conv_output_size(
          input.sizes(),
          weight.sizes(),
          padding.vec(),
          stride.vec(),
          {1, 1},
          groups),
      device(kDPCPP).dtype(kQUInt8),
      output_scale,
      output_zero_point,
      MemoryFormat::Contiguous);
  output = convolution(
      output,
      input,
      weight,
      bias,
      padding.vec(),
      stride.vec(),
      {1, 1},
      groups,
      attr);

  return output;
}

static auto registry =
    c10::RegisterOperators()
        .op("quantized::conv2d",
            c10::RegisterOperators::options()
                .kernel<decltype(dpcppConv), &dpcppConv>(
                    DispatchKey::QuantizedDPCPPTensorId))
        .op("quantized::conv2d_relu",
            c10::RegisterOperators::options()
                .kernel<decltype(dpcppConvRelu), &dpcppConvRelu>(
                    DispatchKey::QuantizedDPCPPTensorId));

} // namespace AtenIpexTypeDPCPP
} // namespace at
