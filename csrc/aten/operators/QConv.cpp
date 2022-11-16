#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/PackedParams.h>
#include "ATen/core/interned_strings.h"
#include "c10/core/MemoryFormat.h"
#include "c10/core/ScalarType.h"
#include "comm/ParamUtils.h"

#include <oneDNN/oneDNN.h>
#include <quantized/QUtils.h>
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

template <class T, class = typename std::enable_if<std::is_integral<T>::value>>
T quantize_value(float scale, int64_t zero_point, float value) {
  int64_t qvalue;
  constexpr int64_t qmin = std::numeric_limits<T>::min();
  constexpr int64_t qmax = std::numeric_limits<T>::max();
  float inv_scale = 1.0f / static_cast<float>(scale);
  qvalue = static_cast<int64_t>(zero_point + std::nearbyint(value * inv_scale));
  qvalue = std::max<int64_t>(qvalue, qmin);
  qvalue = std::min<int64_t>(qvalue, qmax);
  return static_cast<T>(qvalue);
}

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
        Tensor(),
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
        Tensor(),
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
    mfmt_ = using_channels_last_for_conv(input, weight_)
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

#define IPEX_QCONV_DEFINATION(op, lambda)                                \
  at::Tensor q_conv2d_##op(                                              \
      const Tensor& input,                                               \
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,  \
      double output_scale,                                               \
      int64_t output_zero_point) {                                       \
    auto qconv_wrapper =                                                 \
        QuantizeConvConverter<2>(packed_weight, 0.00392157, 0, kQUInt8); \
    return qconv_wrapper.call(input, lambda);                            \
  }

} // namespace

namespace at {
namespace AtenIpexTypeQuantizedXPU {

// In QConv op, we use binary_add post-op to implement the functionality of bias
// add. To add bias into QConv op, oneDNN requires to adjust bias value from
// FP32 range to S32 range by multiplying src_scale * weight_scale. Considering
// the performance, we cache the S32 bias value in device memory, which may
// result in in-correct accuracy when users query the bias value in inference
// time, or the bias tensor is also needed by other operator (like int8 JIT save
// case). To achieve peak performance and keep accuracy at the same time, we use
// binary_add post-op to add bias tensor to QConv result. In this case, oneDNN
// performs binary_add post op in FP32 range, no value change is needed for bias
// tensor. Using binary_add to implement bias add is only implemented for QConv,
// QLinear. For the other datatype, we still use bias add and there's no
// performance and accuracy issue. Notice: bias in Conv should be in shape of
// [OC] binary_add post-op in oneDNN needs the binary tensor in shape of
// [1,1,1,1], or [1,OC,1,1], or [N,OC,OH,OW]
// so we need to view bias from [OC] to [1,OC,1,1]

Tensor q_conv2d(
    Tensor input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, output_scale, 0, kQInt8);
  auto post_op = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr;
  };
  return qconv_wrapper.call(input, post_op);
}

at::Tensor q_conv2d_relu(
    Tensor input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, output_scale, 128, kQUInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_relu);
  };
  return qconv_wrapper.call(input, att);
}

Tensor q_conv3d(
    Tensor input,
    const c10::intrusive_ptr<ConvPackedParamsBase<3>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper =
      QuantizeConvConverter<3>(packed_weight, output_scale, 0, kQInt8);
  auto post_op = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<3>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<3>(bias);
    }
    return attr;
  };
  return qconv_wrapper.call(input, post_op);
}

Tensor q_conv3d_relu(
    Tensor input,
    const c10::intrusive_ptr<ConvPackedParamsBase<3>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper =
      QuantizeConvConverter<3>(packed_weight, output_scale, 128, kQUInt8);
  auto post_op = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<3>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<3>(bias);
    }
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

at::Tensor q_conv2d_sum(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Tensor& accumu,
    float sum_scale,
    int sum_zero_point) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, sum_scale, sum_zero_point, kQInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(sum_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_sum(1.f, accumu.q_scale());
  };
  return qconv_wrapper.call(input, accumu, att);
}

at::Tensor q_conv2d_sum_relu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Tensor& accumu,
    float sum_scale,
    int sum_zero_point) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, sum_scale, sum_zero_point, kQUInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(sum_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_sum(1.f, accumu.q_scale())
        .append_post_eltwise(1.f, 0.f, 0.f, attr.kind_with_relu);
  };
  return qconv_wrapper.call(input, accumu, att);
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
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_sigmoid);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_relu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, output_zero_point, ScalarType::QUInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_relu);
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
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
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
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
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
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
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

at::Tensor q_conv2d_sqrt(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, 0.00392157, 0, kQUInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(1.0 / 255.0));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_sqrt);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_abs(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, 0.00392157, 0, kQUInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(1.0 / 255.0));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_abs);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_tanh(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, output_zero_point, kQUInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_tanh);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_square(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, 0.00392157, 0, kQUInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(1.0 / 255.0));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_tanh);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_exp(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, 0.00392157, 0, kQUInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(1.0 / 255.0));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_exp);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_log(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, output_zero_point, kQInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_log);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_round(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, output_zero_point, kQInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_round);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_log_sigmoid(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, output_zero_point, kQInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_logsigmoid);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_hardswish(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, output_zero_point, kQInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_hardswish);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_mish(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, output_zero_point, kQInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_mish);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_silu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, output_zero_point, kQInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 1.f,
        /* beta */ 0.f,
        attr.kind_with_swish);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_gelu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    c10::string_view approximate) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, output_zero_point, kQInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_gelu);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_hardsigmoid(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, 0.00392157, 0, kQUInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(1.0 / 255.0));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 1. / 6.,
        /* beta */ 1. / 2.,
        attr.kind_with_hardsigmoid);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_pow(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Scalar exponent) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, output_zero_point, kQInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      bias = bias.view({1, bias.size(0), 1, 1});
      attr.append_post_binary(attr.kind_with_binary_add, bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 1.f,
        /* beta */ exponent.toFloat(),
        attr.kind_with_pow);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_hardtanh(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Scalar minval,
    Scalar maxval) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, output_zero_point, kQInt8);
  auto att = [=]() {
    int32_t min_q = quantize_value<int32_t>(
        output_scale, output_zero_point, minval.toFloat());
    int32_t max_q = quantize_value<int32_t>(
        output_scale, output_zero_point, maxval.toFloat());
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      bias = bias.view({1, bias.size(0), 1, 1});
      attr.append_post_binary(attr.kind_with_binary_add, bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ static_cast<float>(min_q),
        /* beta */ static_cast<float>(max_q),
        attr.kind_with_clip);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_elu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale) {
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, output_zero_point, kQInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      bias = bias.view({1, bias.size(0), 1, 1});
      attr.append_post_binary(attr.kind_with_binary_add, bias);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ alpha.toFloat(),
        /* beta */ 1.0,
        attr.kind_with_elu);
  };
  return qconv_wrapper.call(input, att);
}

} // namespace AtenIpexTypeXPU
} // namespace at
