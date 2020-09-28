#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <utils/ParamUtils.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>
#include <core/Quantizer.h>
#include "Conv.h"
#include "QUtil.h"

using namespace mkldnn;
using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {

using namespace impl;

at::Tensor q_conv2d(
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

  conv_attr_t attr = {1.f, 0.f, 0.f, output_scale, 0};

  Tensor output = _empty_affine_quantized(
      conv_output_size(
          input.sizes(),
          weight.sizes(),
          padding.vec(),
          stride.vec(),
          dilation.vec(),
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
      dilation.vec(),
      groups,
      attr);

  return output;
}

at::Tensor q_conv2d_relu(
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

  conv_attr_t attr = {1.f, 0.f, 0.f, output_scale, conv_attr_t::kind_with_relu};

  Tensor output = _empty_affine_quantized(
      conv_output_size(
          input.sizes(),
          weight.sizes(),
          padding.vec(),
          stride.vec(),
          dilation.vec(),
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
      dilation.vec(),
      groups,
      attr);

  return output;
}

at::Tensor q_conv2d_sum_relu(
    Tensor& accumu,
    const Tensor& input,
    const Tensor& packed_weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    double conv_scale,
    int64_t conv_zero_point,
    double sum_scale,
    int64_t sum_zero_point) {
  auto& pack_ptr =
      cpp_custom_type_hack::cast<PackedConvWeightQDPCPP>(packed_weight);

  at::Tensor weight = pack_ptr.weight;
  at::Tensor bias;
  if (pack_ptr.bias.has_value())
    bias = pack_ptr.bias.value();

  conv_attr_t attr = {accumu.q_scale() / sum_scale, 0.f, 0.f,
      sum_scale, conv_attr_t::kind_with_relu | conv_attr_t::kind_with_sum};

  convolution(
      accumu,
      input,
      weight,
      bias,
      padding.vec(),
      stride.vec(),
      dilation.vec(),
      groups,
      attr);

  accumu.set_quantizer_(
      at::dpcpp::make_per_tensor_affine_quantizer(
      sum_scale, sum_zero_point, accumu.scalar_type()));

  return accumu;
}

static auto registry =
    c10::RegisterOperators()
        .op("quantized::conv2d",
            c10::RegisterOperators::options()
                .kernel<decltype(q_conv2d), &q_conv2d>(
                    DispatchKey::QuantizedDPCPPTensorId))
        .op("quantized::conv2d_relu",
            c10::RegisterOperators::options()
                .kernel<decltype(q_conv2d_relu), &q_conv2d_relu>(
                    DispatchKey::QuantizedDPCPPTensorId));

} // namespace AtenIpexTypeDPCPP
} // namespace at
