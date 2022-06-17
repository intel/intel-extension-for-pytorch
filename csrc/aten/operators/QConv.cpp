#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/PackedParams.h>
#include "ATen/core/interned_strings.h"
#include "c10/core/MemoryFormat.h"
#include "c10/core/ScalarType.h"
#include "comm/ParamUtils.h"

#include <oneDNN/oneDNN.h>
#include <quantized/QUtil.h>
#include <quantized/Quantizer.h>
#include <runtime/Utils.h>
#include <unistd.h>
#include "comm/RegistrationDeclarations.h"

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace {
// the struct announced below should only be used in this file for the
// defination of quantized convolution with post op implementation.

template <int N>
struct QuantizeConvConverter {
  QuantizeConvConverter(
      c10::intrusive_ptr<ConvPackedParamsBase<N>> packed_weight,
      double q_scale,
      int64_t q_zero_point,
      at::ScalarType type) {
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<N>*>(packed_weight.get());
    weight_ = pack_ptr->weight;
    if (pack_ptr->bias.has_value()) {
      bias_ = pack_ptr->bias.value();
    }
    padding_ = pack_ptr->padding();
    stride_ = pack_ptr->stride();
    groups_ = pack_ptr->groups();
    dilation_ = pack_ptr->dilation();
    q_scale_ = q_scale;
    q_zero_point_ = q_zero_point;
    dtype_ = type;
  }

  template <typename Func>
  at::Tensor call(const at::Tensor& input, Func func) {
    Attr att = func();
    at::Tensor output = quantizedEmptyTensorFromInput(input);
    output = convolution(
        output,
        input,
        weight_,
        bias_,
        padding_.vec(),
        padding_.vec(),
        stride_.vec(),
        dilation_.vec(),
        groups_,
        att);
    return output;
  }

  template <typename Func>
  at::Tensor call(const at::Tensor& input, at::Tensor& output, Func func) {
    Attr att = func();
    output = convolution(
        output,
        input,
        weight_,
        bias_,
        padding_.vec(),
        padding_.vec(),
        stride_.vec(),
        dilation_.vec(),
        groups_,
        att);
    set_quantizer_(
        output,
        dpcpp_make_per_tensor_affine_quantizer(
            q_scale_, q_zero_point_, dtype_));
    return output;
  }

  at::Tensor quantizedEmptyTensorFromInput(const at::Tensor& input) {
    at::MemoryFormat channel_last_fmt;
    switch (N) {
      // ChannelsLast1d flag seems not visible in IPEX pre-ci
      // case 1:
      //   channel_last_fmt = at::MemoryFormat::ChannelsLast1d;
      //   break;
      case 2:
        channel_last_fmt = at::MemoryFormat::ChannelsLast;
        break;
      case 3:
        channel_last_fmt = at::MemoryFormat::ChannelsLast3d;
        break;
      default:
        AT_ERROR(
            "QConv.cpp: IPEX dose not support quantized convolution have dimension more than 3.");
    }
    mfmt_ = onednn_conv_use_channels_last(input, weight_)
        ? channel_last_fmt
        : at::MemoryFormat::Contiguous;
    return at::_empty_affine_quantized(
        conv_dst_tz(
            input.ndimension(),
            input.sizes(),
            weight_.sizes(),
            padding_.vec(),
            padding_.vec(),
            stride_.vec(),
            dilation_.vec()),
        device(at::kXPU).dtype(dtype_),
        q_scale_,
        q_zero_point_,
        mfmt_);
  }
  at::Tensor weight_;
  at::Tensor bias_;
  at::Tensor output_;
  torch::List<int64_t> padding_;
  torch::List<int64_t> stride_;
  int64_t groups_;
  torch::List<int64_t> dilation_;
  at::MemoryFormat channel_last_format_;
  at::MemoryFormat mfmt_;
  double q_scale_;
  int64_t q_zero_point_;
  at::ScalarType dtype_;
};
} // namespace

namespace at {
namespace AtenIpexTypeQuantizedXPU {

Tensor q_conv2d(
    Tensor input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, output_zero_point, kQInt8);
  auto post_op = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    return attr;
  };
  return qconv_wrapper.call(input, post_op);
}

Tensor q_conv2d_relu(
    Tensor input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, output_zero_point, kQUInt8);
  auto post_op = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_relu);
  };
  return qconv_wrapper.call(input, post_op);
}

Tensor q_conv3d(
    Tensor input,
    const c10::intrusive_ptr<ConvPackedParamsBase<3>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper = QuantizeConvConverter<3>(
      packed_weight, output_scale, output_zero_point, kQInt8);
  auto post_op = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    return attr;
  };
  return qconv_wrapper.call(input, post_op);
}

Tensor q_conv3d_relu(
    Tensor input,
    const c10::intrusive_ptr<ConvPackedParamsBase<3>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper = QuantizeConvConverter<3>(
      packed_weight, output_scale, output_zero_point, kQUInt8);
  auto post_op = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_relu);
  };
  return qconv_wrapper.call(input, post_op);
}

TORCH_LIBRARY_IMPL(quantized, QuantizedXPU, m) {
  m.impl("quantized::conv2d.new", q_conv2d);
  m.impl("quantized::conv2d_relu.new", q_conv2d_relu);
  m.impl("quantized::conv3d.new", q_conv3d);
  m.impl("quantized::conv3d_relu.new", q_conv3d_relu);
}

} // namespace AtenIpexTypeQuantizedXPU

namespace AtenIpexTypeXPU {
// haven't been covered by ut
at::Tensor q_conv2d_sum_relu(
    Tensor& accumu,
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double q_conv_scale,
    int64_t q_conv_zero_point,
    double q_sum_scale,
    int64_t q_sum_zero_point) {
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

  // output = eltwise_scale * Relu(conv_scale * Conv(input, weight) + sum_scale
  // * accumu)

  // since the input/output for RELU are share the same scale,
  // then the fused op q_scale = q_sum_scale
  Attr attr(/* q_scale */ static_cast<float>(q_sum_scale));
  attr.append_post_sum(/*sum_scale*/ 1.f, /* sum_q_scale */ accumu.q_scale());
  attr.append_post_eltwise(
      /* eltwise_scale */ 1.f,
      /* alpha */ 0.f,
      /* beta */ 0.f,
      attr.kind_with_relu);

  convolution(
      accumu,
      input,
      weight,
      bias,
      padding.vec(),
      padding.vec(),
      stride.vec(),
      dilation.vec(),
      groups,
      attr);

  set_quantizer_(
      accumu,
      dpcpp_make_per_tensor_affine_quantizer(
          q_sum_scale, q_sum_zero_point, accumu.scalar_type()));

  return accumu;
}

at::Tensor q_conv2d_sigmoid(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, 0.00392157, 0, kQUInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(1.0 / 255.0));
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_sigmoid);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_leaky_relu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Scalar negative_slope) {
  float alpha = negative_slope.to<float>();
  auto dtype = alpha <= 0.0 ? ScalarType::QUInt8 : ScalarType::QInt8;
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, output_zero_point, dtype);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ alpha,
        /* beta */ 0.f,
        attr.kind_with_relu);
  };
  return qconv_wrapper.call(input, att);
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
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, q_scale, q_zpoint, kQInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(q_scale));
    return attr.append_post_eltwise(
        /* mish_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_mish);
  };
  return qconv_wrapper.call(input, att);
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
    Tensor accumu,
    double add_scale,
    int64_t add_zero_point) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, add_scale, add_zero_point, accumu.scalar_type());
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(add_scale));
    return attr
        .append_post_eltwise(
            /* mish_scale */ 1.f,
            /* alpha */ 0.f,
            /* beta */ 0.f,
            attr.kind_with_mish)
        .append_post_sum(
            /* sum_scale */ 1.f,
            /* sum_q_scale */ accumu.q_scale());
  };
  return qconv_wrapper.call(input, accumu, att);
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
