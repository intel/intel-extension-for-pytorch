#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include "comm/ParamUtils.h"

#include <oneDNN/oneDNN.h>
#include <quantized/QUtil.h>
#include <quantized/Quantizer.h>
#include <runtime/Utils.h>
#include "comm/RegistrationDeclarations.h"

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeQuantizedXPU {

at::Tensor q_conv2d(
    Tensor input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto pack_ptr = dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());

  at::Tensor weight = pack_ptr->weight;
  at::Tensor bias;
  if (pack_ptr->bias.has_value())
    bias = pack_ptr->bias.value();
  auto padding = pack_ptr->padding();
  auto stride = pack_ptr->stride();
  auto groups = pack_ptr->groups();
  auto dilation = pack_ptr->dilation();

  ConvAttr attr = {1.f, 0.f, 0.f, static_cast<float>(output_scale), 0};

  auto mfmt = onednn_conv_use_channels_last(input, weight)
      ? at::MemoryFormat::ChannelsLast
      : at::MemoryFormat::Contiguous;

  Tensor output = _empty_affine_quantized(
      conv_dst_tz(
          input.ndimension(),
          input.sizes(),
          weight.sizes(),
          padding.vec(),
          stride.vec(),
          dilation.vec()),
      device(kXPU).dtype(kQInt8),
      output_scale,
      output_zero_point,
      mfmt);

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
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  return at::AtenIpexTypeXPU::q_conv2d_leaky_relu(
      input, packed_weight, output_scale, output_zero_point, Scalar(0.0));
}

at::Tensor q_conv3d(
    Tensor input,
    const c10::intrusive_ptr<ConvPackedParamsBase<3>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto pack_ptr = dynamic_cast<PackedConvWeightQDPCPP<3>*>(packed_weight.get());

  at::Tensor weight = pack_ptr->weight;
  at::Tensor bias;
  if (pack_ptr->bias.has_value())
    bias = pack_ptr->bias.value();
  auto padding = pack_ptr->padding();
  auto stride = pack_ptr->stride();
  auto groups = pack_ptr->groups();
  auto dilation = pack_ptr->dilation();

  ConvAttr attr = {1.f, 0.f, 0.f, static_cast<float>(output_scale), 0};

  auto mfmt = onednn_conv_use_channels_last(input, weight)
      ? at::MemoryFormat::ChannelsLast3d
      : at::MemoryFormat::Contiguous;
  Tensor output = _empty_affine_quantized(
      conv_dst_tz(
          input.ndimension(),
          input.sizes(),
          weight.sizes(),
          padding.vec(),
          stride.vec(),
          dilation.vec()),
      device(kXPU).dtype(kQInt8),
      output_scale,
      output_zero_point,
      mfmt);

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

at::Tensor q_conv3d_relu(
    Tensor input,
    const c10::intrusive_ptr<ConvPackedParamsBase<3>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto pack_ptr = dynamic_cast<PackedConvWeightQDPCPP<3>*>(packed_weight.get());

  at::Tensor weight = pack_ptr->weight;
  at::Tensor bias;

  if (pack_ptr->bias.has_value())
    bias = pack_ptr->bias.value();
  auto padding = pack_ptr->padding();
  auto stride = pack_ptr->stride();
  auto groups = pack_ptr->groups();
  auto dilation = pack_ptr->dilation();

  ConvAttr attr = {
      1.f,
      0.f,
      0.f,
      static_cast<float>(output_scale),
      ConvAttr::kind_with_relu};

  auto mfmt = onednn_conv_use_channels_last(input, weight)
      ? at::MemoryFormat::ChannelsLast3d
      : at::MemoryFormat::Contiguous;

  Tensor output = _empty_affine_quantized(
      conv_dst_tz(
          input.ndimension(),
          input.sizes(),
          weight.sizes(),
          padding.vec(),
          stride.vec(),
          dilation.vec()),
      device(kXPU).dtype(kQUInt8),
      output_scale,
      output_zero_point,
      mfmt);

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

TORCH_LIBRARY_IMPL(quantized, QuantizedXPU, m) {
  m.impl("quantized::conv2d.new", q_conv2d);
  m.impl("quantized::conv2d_relu.new", q_conv2d_relu);
  m.impl("quantized::conv3d.new", q_conv3d);
  m.impl("quantized::conv3d_relu.new", q_conv3d_relu);
}

} // namespace AtenIpexTypeQuantizedXPU

namespace AtenIpexTypeXPU {
at::Tensor q_conv2d_sum_relu(
    Tensor& accumu,
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double conv_scale,
    int64_t conv_zero_point,
    double sum_scale,
    int64_t sum_zero_point) {
  auto pack_ptr =
      dynamic_cast<AtenIpexTypeQuantizedXPU::PackedConvWeightQDPCPP<2>*>(
          packed_weight.get());

  at::Tensor weight = pack_ptr->weight;
  at::Tensor bias;
  if (pack_ptr->bias.has_value())
    bias = pack_ptr->bias.value();
  auto padding = pack_ptr->padding();
  auto stride = pack_ptr->stride();
  auto groups = pack_ptr->groups();
  auto dilation = pack_ptr->dilation();

  ConvAttr attr = {
      static_cast<float>(accumu.q_scale() / sum_scale),
      0.f,
      0.f,
      static_cast<float>(sum_scale),
      ConvAttr::kind_with_relu | ConvAttr::kind_with_sum};

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

  set_quantizer_(
      accumu,
      dpcpp_make_per_tensor_affine_quantizer(
          sum_scale, sum_zero_point, accumu.scalar_type()));

  return accumu;
}

at::Tensor q_conv2d_leaky_relu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Scalar negative_slope) {
  auto pack_ptr = dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());

  at::Tensor weight = pack_ptr->weight;
  at::Tensor bias;
  if (pack_ptr->bias.has_value())
    bias = pack_ptr->bias.value();
  auto padding = pack_ptr->padding();
  auto stride = pack_ptr->stride();
  auto groups = pack_ptr->groups();
  auto dilation = pack_ptr->dilation();

  float alpha = negative_slope.to<float>();
  ConvAttr attr = {
      1.f,
      alpha,
      0.f,
      static_cast<float>(output_scale),
      ConvAttr::kind_with_relu};

  auto mfmt = onednn_conv_use_channels_last(input, weight)
      ? at::MemoryFormat::ChannelsLast
      : at::MemoryFormat::Contiguous;

  Tensor output = at::_empty_affine_quantized(
      conv_dst_tz(
          input.ndimension(),
          input.sizes(),
          weight.sizes(),
          padding.vec(),
          stride.vec(),
          dilation.vec()),
      device(kXPU).dtype(kQUInt8),
      output_scale,
      output_zero_point,
      mfmt);

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

at::Tensor q_conv2d_dequantize(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  Tensor qconv_output = at::AtenIpexTypeQuantizedXPU::q_conv2d(
      input, packed_weight, output_scale, output_zero_point);
  Tensor dequantized_output = at::dequantize(qconv_output);
  return dequantized_output;
}

at::Tensor q_conv2d_dequantize_softplus_tanh_mul_quantize(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    const Scalar& beta,
    const Scalar& threshold,
    double q_scale,
    int64_t q_zpoint,
    at::ScalarType dtype) {
  auto pack_ptr = dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());

  at::Tensor weight = pack_ptr->weight;
  at::Tensor bias;
  if (pack_ptr->bias.has_value())
    bias = pack_ptr->bias.value();
  auto padding = pack_ptr->padding();
  auto stride = pack_ptr->stride();
  auto groups = pack_ptr->groups();
  auto dilation = pack_ptr->dilation();

  ConvAttr attr = {
      1.f,
      0.f,
      0.f,
      static_cast<float>(output_scale),
      ConvAttr::kind_with_mish};

  auto mfmt = onednn_conv_use_channels_last(input, weight)
      ? at::MemoryFormat::ChannelsLast
      : at::MemoryFormat::Contiguous;

  Tensor output = at::_empty_affine_quantized(
      conv_dst_tz(
          input.ndimension(),
          input.sizes(),
          weight.sizes(),
          padding.vec(),
          stride.vec(),
          dilation.vec()),
      device(kXPU).dtype(kQUInt8),
      output_scale,
      output_zero_point,
      mfmt);

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

at::Tensor q_conv2d_dequantize_softplus_tanh_mul_quantize_add(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    const Scalar& beta,
    const Scalar& threshold,
    double q_scale,
    int64_t q_zpoint,
    at::ScalarType dtype,
    Tensor qb,
    double add_scale,
    int64_t add_zero_point) {
  auto pack_ptr = dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());

  at::Tensor weight = pack_ptr->weight;
  at::Tensor bias;
  if (pack_ptr->bias.has_value())
    bias = pack_ptr->bias.value();
  auto padding = pack_ptr->padding();
  auto stride = pack_ptr->stride();
  auto groups = pack_ptr->groups();
  auto dilation = pack_ptr->dilation();

  ConvAttr attr = {
      1.f,
      0.f,
      0.f,
      static_cast<float>(output_scale),
      ConvAttr::kind_with_mish | ConvAttr::kind_with_linear};

  auto mfmt = onednn_conv_use_channels_last(input, weight)
      ? at::MemoryFormat::ChannelsLast
      : at::MemoryFormat::Contiguous;

  Tensor output = at::_empty_affine_quantized(
      conv_dst_tz(
          input.ndimension(),
          input.sizes(),
          weight.sizes(),
          padding.vec(),
          stride.vec(),
          dilation.vec()),
      device(kXPU).dtype(kQUInt8),
      output_scale,
      output_zero_point,
      mfmt);

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

Tensor softplus_tanh(
    const Tensor& self,
    const Scalar& beta,
    const Scalar& threshold) {
  Tensor softplus_out = at::AtenIpexTypeXPU::softplus(self, beta, threshold);
  return at::tanh(softplus_out);
}

Tensor softplus_tanh_mul(
    const Tensor& self,
    const Scalar& beta,
    const Scalar& threshold,
    const Tensor& mul_input) {
  return at::mul(
      self, at::AtenIpexTypeXPU::softplus_tanh(self, beta, threshold));
}

Tensor q_conv2d_dequantize_softplus_tanh_mul(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    const Scalar& beta,
    const Scalar& threshold) {
  Tensor dequantize_out = at::AtenIpexTypeXPU::q_conv2d_dequantize(
      input, packed_weight, output_scale, output_zero_point);
  return at::AtenIpexTypeXPU::softplus_tanh_mul(
      dequantize_out, beta, threshold, input);
}

} // namespace AtenIpexTypeXPU
} // namespace at
