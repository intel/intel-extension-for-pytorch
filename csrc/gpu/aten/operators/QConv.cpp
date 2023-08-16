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
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include "comm/RegistrationDeclarations.h"
#include "utils/CustomOperatorRegistration.h"

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
    // [Note: Method to set fusion output dtype]
    // Considering two calling methods in QuantizedConvConverter,
    // 1. wrapper.call(input, func): the output is created in
    //  QuantizedConvConverter, we may choose u8 for conv_xxx_relu fusion,
    //  while s8 for conv_xxx_non_relu. Please use 128 as zp for u8,
    //  while 0 for s8.
    // 2. wrapper.call(input, output, func): output is created out of
    //   QuantizedConvConverter, its scalar_type should not be changed.
    //   For conv_xxx_relu, please use output.scalar_type() for constructing
    //   Converter, rather than u8. Otherwise, it may results in a tensor's
    //   quantizer has different scalar type as tensor.
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<N>*>(packed_weight.get());
    weight_ = pack_ptr->weight;
    padding_ = pack_ptr->padding();
    stride_ = pack_ptr->stride();
    groups_ = pack_ptr->groups();
    dilation_ = pack_ptr->dilation();
    transpose_ = pack_ptr->transpose();
    output_padding_ = pack_ptr->output_padding();
    transpose_ = pack_ptr->transpose();
    q_scale_ = q_scale;
    q_zero_point_ = q_zero_point;
    dtype_ = type;
  }

  template <typename Func>
  at::Tensor call(const at::Tensor& input, Func func) {
    // make sure input/weight/output are contiguous or ChannelsLast congituous
    at::MemoryFormat mfmt = get_tensor_format_for_conv(input, weight_);
    Tensor input_ = is_onednn_layout(input) ? input.contiguous(mfmt) : input;
    weight_ = is_onednn_layout(weight_) ? weight_.contiguous(mfmt) : weight_;
    at::Tensor output_ = quantizedEmptyTensorFromInput(input_);

    Attr att = func();
    if (!transpose_) {
      output_ = quantized_convolution(
          output_,
          input_,
          weight_,
          padding_.vec(),
          padding_.vec(),
          stride_.vec(),
          dilation_.vec(),
          groups_,
          att);
    } else {
      output_ = quantized_deconvolution(
          output_,
          input_,
          weight_,
          Tensor(),
          stride_.vec(),
          padding_.vec(),
          output_padding_.vec(),
          dilation_.vec(),
          groups_);
    }
    return output_;
  }

  template <typename Func>
  at::Tensor call(const at::Tensor& input, at::Tensor& output, Func func) {
    // make sure input/weight are contiguous or ChannelsLast congituous
    at::MemoryFormat mfmt = get_tensor_format_for_conv(input, weight_);
    Tensor input_ = is_onednn_layout(input) ? input.contiguous(mfmt) : input;
    weight_ = is_onednn_layout(weight_) ? weight_.contiguous(mfmt) : weight_;
    Tensor output_ = output.is_contiguous(mfmt)
        ? output
        : quantizedEmptyTensorFromInput(input_);

    Attr att = func();
    if (!transpose_) {
      output = quantized_convolution(
          output_,
          input_,
          weight_,
          padding_.vec(),
          padding_.vec(),
          stride_.vec(),
          dilation_.vec(),
          groups_,
          att);
      if (!output.is_same(output_)) {
        output.copy_(output_);
      }
      set_quantizer_(
          output,
          dpcpp_make_per_tensor_affine_quantizer(
              q_scale_, q_zero_point_, dtype_));
    } else {
      output_ = quantized_deconvolution(
          output_,
          input_,
          weight_,
          Tensor(),
          stride_.vec(),
          padding_.vec(),
          output_padding_.vec(),
          dilation_.vec(),
          groups_);

      set_quantizer_(
          output_,
          dpcpp_make_per_tensor_affine_quantizer(
              q_scale_, q_zero_point_, dtype_));
    }
    return output_;
  }

  at::Tensor quantizedEmptyTensorFromInput(const at::Tensor& input) {
    auto dst_tz = transpose_ ? deconv_dst_tz(
                                   input.sizes(),
                                   weight_.sizes(),
                                   padding_.vec(),
                                   stride_.vec(),
                                   dilation_.vec(),
                                   output_padding_.vec(),
                                   groups_)
                             : conv_dst_tz(
                                   input.ndimension(),
                                   input.sizes(),
                                   weight_.sizes(),
                                   padding_.vec(),
                                   padding_.vec(),
                                   stride_.vec(),
                                   dilation_.vec());
    return at::_empty_affine_quantized(
        dst_tz,
        device(at::kXPU).dtype(dtype_),
        q_scale_,
        q_zero_point_,
        input.suggest_memory_format());
  }
  at::Tensor weight_;
  at::Tensor output_;
  torch::List<int64_t> padding_;
  torch::List<int64_t> stride_;
  torch::List<int64_t> output_padding_;
  int64_t groups_;
  bool transpose_;
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
  // See [Note: Method to set fusion output dtype]
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
  // See [Note: Method to set fusion output dtype]
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
  // See [Note: Method to set fusion output dtype]
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
  // See [Note: Method to set fusion output dtype]
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

Tensor q_deconv3d(
    Tensor input,
    const c10::intrusive_ptr<ConvPackedParamsBase<3>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<3>(packed_weight, output_scale, 0, kQInt8);

  // TODO: no post op for q_deconv3d
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

TORCH_LIBRARY_IMPL(quantized, QuantizedXPU, m) {
  IPEX_QOP_REGISTER("quantized::conv2d.new", q_conv2d);
  IPEX_QOP_REGISTER("quantized::conv2d_relu.new", q_conv2d_relu)
  IPEX_QOP_REGISTER("quantized::conv3d.new", q_conv3d);
  IPEX_QOP_REGISTER("quantized::conv3d_relu.new", q_conv3d_relu);
  IPEX_QOP_REGISTER("quantized::conv_transpose3d", q_deconv3d);
}

} // namespace AtenIpexTypeQuantizedXPU

namespace AtenIpexTypeXPU {

at::Tensor q_conv2d_sum(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Tensor& accumu,
    double sum_scale,
    int64_t sum_zero_point) {
  // See [Note: Method to set fusion output dtype]
  auto output_zp = accumu.scalar_type() == kQUInt8 ? 128 : 0;
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, sum_scale, output_zp, accumu.scalar_type());
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(sum_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr.append_post_sum(
        1.f,
        accumu.scalar_type() == kQUInt8
            ? accumu.q_scale() / 2
            : accumu.q_scale()); // See [Note: Gap of u8 qtensor scale between
                                 // oneDNN and PyTorch]
  };
  return qconv_wrapper.call(input, accumu, att);
}

at::Tensor q_conv2d_sum_relu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Tensor& accumu,
    double sum_scale,
    int64_t sum_zero_point) {
  // See [Note: Method to set fusion output dtype]
  auto output_zp = accumu.scalar_type() == kQUInt8 ? 128 : 0;
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, sum_scale, output_zp, accumu.scalar_type());
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(sum_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    return attr
        .append_post_sum(
            1.f,
            accumu.scalar_type() == kQUInt8
                ? accumu.q_scale() / 2
                : accumu.q_scale()) // See [Note: Gap of u8 qtensor scale
                                    // between oneDNN and PyTorch]
        .append_post_eltwise(1.f, 0.f, 0.f, attr.kind_with_relu);
  };
  return qconv_wrapper.call(input, accumu, att);
}

at::Tensor q_conv2d_sigmoid(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  // [Note: Scale caching for qsigmoid]
  // PyTorch qsigmoid uses 1.0 / 256.0 as qscale for qsigmoid rather than
  // get scale from observer. scale&zp caching in IPEX caches torch_scale / 2
  // for dnn symmetric qunt diff with PyTorch. To cache right value for qsimoid
  // here, we let scale = 1.0 / 255.0 * 2.0 The output tensor would has
  // observed-like scale, while it caches right scale inside.
  auto scale = 1.0 / 255.0 * 2.0;
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, scale, 128, kQUInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(scale));
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
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, output_scale, 128, ScalarType::QUInt8);
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
  // See [Note: Method to set fusion output dtype]
  float alpha = negative_slope.to<float>();
  auto dtype = alpha <= 0.0 ? ScalarType::QUInt8 : ScalarType::QInt8;
  auto output_zp = (dtype == kQUInt8) ? 128 : 0;
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, output_scale, output_zp, dtype);
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

at::Tensor q_conv2d_mish_compound(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    const Scalar& beta,
    const Scalar& threshold,
    double q_scale,
    int64_t q_zpoint,
    at::ScalarType dtype) {
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, q_scale, 0, kQInt8);
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

at::Tensor q_conv2d_mish_compound_add(
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
  // See [Note: Method to set fusion output dtype]
  auto output_zp = accumu.scalar_type() == kQUInt8 ? 128 : 0;
  auto qconv_wrapper = QuantizeConvConverter<2>(
      packed_weight, add_scale, output_zp, accumu.scalar_type());
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

Tensor mish_compound(
    const Tensor& self,
    const Scalar& beta,
    const Scalar& threshold,
    const Tensor& mul_input) {
  return at::mul(
      self, at::AtenIpexTypeXPU::softplus_tanh(self, beta, threshold));
}

Tensor q_conv2d_dequantize_mish_compound(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    const Scalar& beta,
    const Scalar& threshold) {
  Tensor dequantize_out = at::AtenIpexTypeXPU::q_conv2d_dequantize(
      input, packed_weight, output_scale, output_zero_point);
  return at::AtenIpexTypeXPU::mish_compound(
      dequantize_out, beta, threshold, input);
}

Tensor q_conv2d_dequantize_silu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  Tensor dequantize_out = at::AtenIpexTypeXPU::q_conv2d_dequantize(
      input, packed_weight, output_scale, output_zero_point);
  return at::silu(dequantize_out);
}

Tensor q_conv2d_dequantize_silu_quantize(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    double q_scale,
    int64_t q_zpoint,
    at::ScalarType dtype) {
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, q_scale, 0, kQInt8);
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
        attr.kind_with_swish);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_sqrt(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, 0.00392157, 128, kQUInt8);
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
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, 0.00392157, 128, kQUInt8);
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
  // See [Note: Method to set fusion output dtype]
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
        attr.kind_with_tanh);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_square(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, 0.00392157, 128, kQUInt8);
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
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, 0.00392157, 128, kQUInt8);
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
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, output_scale, 0, kQInt8);
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
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, output_scale, 0, kQInt8);
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
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, output_scale, 0, kQInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    // logsigmoid will be removed. It can be used as current soft_relu_v2 with
    // alpha equal to -1. Notice: soft_relu_v2 will be called soft_relu
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ -1.f,
        /* beta */ 0.f,
        attr.kind_with_soft_relu);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_hardswish(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, output_scale, 0, kQInt8);
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
        /* alpha */ 1.0f / 6.0f,
        /* beta */ 1.0f / 2.0f,
        attr.kind_with_hardswish);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_mish(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, output_scale, 0, kQInt8);
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
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, output_scale, 0, kQInt8);
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
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, output_scale, 0, kQInt8);
  auto att = [=]() {
    Attr attr(/* q_scale */ static_cast<float>(output_scale));
    auto pack_ptr =
        dynamic_cast<PackedConvWeightQDPCPP<2>*>(packed_weight.get());
    if (pack_ptr->bias.has_value()) {
      Tensor bias = pack_ptr->bias.value();
      attr.append_bias<2>(bias);
    }
    algorithm algo;
    if (approximate == "none") {
      algo = attr.kind_with_gelu_erf;
    } else if (approximate == "tanh") {
      algo = attr.kind_with_gelu_tanh;
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unsupported gelu algorithm: ", approximate);
    }
    return attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        algo);
  };
  return qconv_wrapper.call(input, att);
}

at::Tensor q_conv2d_hardsigmoid(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, 0.00392157, 128, kQUInt8);
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
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, output_scale, 0, kQInt8);
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
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, output_scale, 0, kQInt8);
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
  // See [Note: Method to set fusion output dtype]
  auto qconv_wrapper =
      QuantizeConvConverter<2>(packed_weight, output_scale, 0, kQInt8);
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

#define IPEX_OP_REGISTER_QCONV(op) \
  IPEX_OP_REGISTER("q_conv2d_" #op, q_conv2d_##op);

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_QCONV(sigmoid);
  IPEX_OP_REGISTER_QCONV(relu);
  IPEX_OP_REGISTER_QCONV(sqrt);
  IPEX_OP_REGISTER_QCONV(abs);
  IPEX_OP_REGISTER_QCONV(tanh);
  IPEX_OP_REGISTER_QCONV(square);
  IPEX_OP_REGISTER_QCONV(exp);
  IPEX_OP_REGISTER_QCONV(log);
  IPEX_OP_REGISTER_QCONV(round);
  IPEX_OP_REGISTER_QCONV(log_sigmoid);
  IPEX_OP_REGISTER_QCONV(hardswish);
  IPEX_OP_REGISTER_QCONV(mish);
  IPEX_OP_REGISTER_QCONV(silu);
  IPEX_OP_REGISTER_QCONV(hardsigmoid);
  IPEX_OP_REGISTER_QCONV(gelu);
  IPEX_OP_REGISTER_QCONV(leaky_relu);
  IPEX_OP_REGISTER_QCONV(pow);
  IPEX_OP_REGISTER_QCONV(hardtanh);
  IPEX_OP_REGISTER_QCONV(elu);
  IPEX_OP_REGISTER_QCONV(sum);
  IPEX_OP_REGISTER_QCONV(sum_relu);
  IPEX_OP_REGISTER_QCONV(dequantize);
  IPEX_OP_REGISTER_QCONV(dequantize_mish_compound);
  IPEX_OP_REGISTER_QCONV(mish_compound);
  IPEX_OP_REGISTER_QCONV(mish_compound_add);
  IPEX_OP_REGISTER_QCONV(dequantize_silu);
  IPEX_OP_REGISTER_QCONV(dequantize_silu_quantize);
  IPEX_OP_REGISTER_NEED_PLAIN("softplus_tanh", softplus_tanh);
  IPEX_OP_REGISTER_NEED_PLAIN("mish_compound", mish_compound)
}

} // namespace AtenIpexTypeXPU
} // namespace at
