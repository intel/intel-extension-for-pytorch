#ifdef USE_LIBXSMM
#include "LinearWoqPacked.h"
#include <ideep.hpp>
#include "aten/Linear.h"
#include "aten/WeightPack.h"
#include "ideep/IDeepConversions.h"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace woq_linear {

c10::intrusive_ptr<WoqLinearOpContext> createWoqLinearPrePackOpContext(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    c10::optional<int64_t> batch_size,
    int64_t lowp_mode,
    int64_t num_concats) {
  RECORD_FUNCTION(
      "ipex_prepack::createWoqLinearPrePackOpContext",
      c10::ArrayRef<c10::IValue>({}));

  return IpexWoqLinearOpContext::create_context(
      std::move(weight), std::move(bias), batch_size, lowp_mode, num_concats);
}

c10::intrusive_ptr<WoqLinearOpContext> createWoqLinearPrePackOpContextInt4(
    at::Tensor&& weight,
    at::Tensor&& scales,
    at::Tensor&& zero_points,
    c10::optional<at::Tensor>&& bias,
    c10::optional<int64_t> batch_size,
    int64_t lowp_mode,
    int64_t num_concats) {
  RECORD_FUNCTION(
      "ipex_prepack::createWoqLinearPrePackOpContextInt4",
      c10::ArrayRef<c10::IValue>({}));
  // From
  // Weight dtype = int32 (uint4 * 8), scale dtype = fp16, zero points dtype =
  // int32 (int4 * 8) To Weight dtype = quint4x2, scale dtype = fp32, zero
  // points dtype = fp32 There might be an extra output channel in weight and
  // scales bool extra_o_channel = false; // scales.numel() >
  // zero_points.numel() * 8;
  auto scales_fp32 = scales.squeeze(0).to(c10::ScalarType::Float);

  // Convert compressed zero points to float
  auto zp_fp32 = at::empty_like(scales_fp32);
  assert(zp_fp32.numel() == zero_points.numel() * 8);
  float* zp_fp32_ptr = reinterpret_cast<float*>(zp_fp32.data_ptr());
  int32_t* zp_int32_ptr = reinterpret_cast<int32_t*>(zero_points.data_ptr());
  for (size_t i = 0; i < zero_points.numel(); ++i) {
    int32_t zp_uint4x8 = zp_int32_ptr[i];
    for (size_t j = 0; j < 8; ++j) {
      zp_fp32_ptr[i * 8 + j] = (float)((zp_uint4x8 >> (j * 4)) & 0xf);
    }
  }
  // Support two cases here:
  // 1. bf16 weight after calibration
  // 2. int4 weight after calibration, quantized and compressed, as int32
  at::Tensor weight_int4;
  if (weight.scalar_type() == c10::kInt) {
    // Weight created by GPTQ and transposed
    // Create empty weight with desired options then copy data
    int64_t N = weight.size(1);
    int64_t K_int32 = weight.size(0);
    int64_t K = K_int32 * 8; // int32 = int4 * 8
    std::vector<int64_t> weight_size = {N, K};
    // Create an empty quint4x2 weight with scales and zero points
    weight_int4 = at::_empty_per_channel_affine_quantized(
        weight_size,
        scales_fp32,
        zp_fp32,
        0,
        device(c10::kCPU).dtype(c10::kQUInt4x2));
    auto weight_t = weight.t().contiguous();
    std::memcpy(
        weight_int4.data_ptr(),
        weight_t.data_ptr(),
        weight_t.numel() * sizeof(uint32_t));
  } else if (weight.scalar_type() == c10::kBFloat16) {
    // Load bf16 weight and quantize
    auto weight_fp32 = weight.to(c10::kFloat);
    weight_int4 = at::quantize_per_channel(
        weight_fp32, scales_fp32, zp_fp32, 0, c10::kQUInt4x2);
  } else if (weight.scalar_type() == c10::kFloat) {
    weight_int4 = at::quantize_per_channel(
        weight, scales_fp32, zp_fp32, 0, c10::kQUInt4x2);
  }
  return IpexWoqLinearOpContext::create_context(
      std::move(weight_int4),
      std::move(bias),
      batch_size,
      lowp_mode,
      num_concats);
}

at::Tensor woq_linear_run(
    const at::Tensor& input,
    c10::intrusive_ptr<WoqLinearOpContext> op_context) {
  RECORD_FUNCTION(
      "ipex_prepack::woq_linear_run", c10::ArrayRef<c10::IValue>({}));

  return op_context->run(input);
}

ContextLinearWoq create(
    at::Tensor& weight,
    at::Tensor& scales,
    at::Tensor& zero_points,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<int64_t> batch_size,
    int64_t lowp_mode,
    int64_t num_concats) {
  auto packed_weight =
      woq_linear_pack_weight(weight, scales, zero_points, lowp_mode);
  bool is_int4 = weight.scalar_type() == c10::kQUInt4x2;
  auto packed_shape = packed_weight.sizes();
  int64_t N = weight.size(0);
  int64_t K = weight.size(1);
  bool weight_is_padded = (packed_shape.size() == 4 && is_int4 &&
                           packed_shape[0] * packed_shape[3] * 2 != N) ||
      (packed_shape.size() == 4 && !is_int4 &&
       packed_shape[0] * packed_shape[3] != N) ||
      (packed_shape.size() == 2 && packed_shape[0] != N);
  auto zero_points_float = zero_points.to(c10::kFloat);
  if (weight_is_padded) {
    int64_t padded_N = packed_shape.size() == 4
        ? (is_int4 ? packed_shape[0] * packed_shape[3] * 2
                   : packed_shape[0] * packed_shape[3])
        : packed_shape[0];
    auto scales_padded = at::pad(scales, {0, padded_N - N}, "constant", 1.f);
    auto zero_points_padded =
        at::pad(zero_points_float, {0, padded_N - N}, "constant", 0.f);
    if (bias.has_value()) {
      auto bias_padded =
          at::pad(bias.value(), {0, padded_N - N}, "constant", 0.f);
      return ContextLinearWoq(
          std::move(packed_weight),
          std::move(scales_padded),
          std::move(zero_points_padded),
          c10::make_optional(bias_padded),
          is_int4,
          lowp_mode,
          num_concats,
          c10::make_optional(weight.sizes().vec()));
    } else {
      return ContextLinearWoq(
          std::move(packed_weight),
          std::move(scales_padded),
          std::move(zero_points_padded),
          c10::nullopt,
          is_int4,
          lowp_mode,
          num_concats,
          c10::make_optional(weight.sizes().vec()));
    }
  }
  return ContextLinearWoq(
      std::move(packed_weight),
      std::move(scales),
      std::move(zero_points_float),
      bias.has_value() ? c10::make_optional(*bias) : c10::nullopt,
      is_int4,
      lowp_mode,
      num_concats,
      weight_is_padded ? c10::make_optional(weight.sizes().vec())
                       : c10::nullopt);
}

at::Tensor run(ContextLinearWoq& context, const at::Tensor& input) {
  // TPP kernel packs weight to 4d (Nc, Kc, block_k, block_n)
  auto w_k = context.at_weight_.dim() == 2
      ? context.at_weight_.size(1)
      : context.at_weight_.size(1) * context.at_weight_.size(2);
  TORCH_CHECK(
      input.size(input.dim() - 1) == w_k,
      "WOQ linear: input and weight shapes do not match, got k = ",
      input.size(input.dim() - 1),
      " and ",
      w_k,
      " respectively.");
  auto input_ = input.contiguous();
  // if weight is not padded, context.orig_wei_shape_ has no value
  if (context.orig_wei_shape_.has_value()) {
    auto res = woq_linear_kernel(
        input_,
        context.at_weight_,
        context.scales_list_,
        context.zero_points_list_,
        context.bias_list_,
        context.is_int4_,
        context.lowp_mode_,
        context.num_concats_);
    // weight shape is [N by K], output shape is [M by N] or [batch by M by N]
    int64_t N = context.orig_wei_shape_.value()[0];
    return at::slice(res, /*dim*/ -1, /*start*/ 0, /*end*/ N, /*step*/ 1);
  }
  return woq_linear_kernel(
      input_,
      context.at_weight_,
      context.scales_list_,
      context.zero_points_list_,
      context.bias_list_,
      context.is_int4_,
      context.lowp_mode_,
      context.num_concats_);
}

// Called by IpexWoqLinearOpContext::run_eltwise
at::Tensor run_eltwise(
    ContextLinearWoq& context,
    const at::Tensor& input,
    const c10::string_view& post_op,
    const torch::List<c10::optional<at::Scalar>>& scalars,
    const c10::optional<c10::string_view>& algorithm) {
  // TPP kernel packs weight to 4d (Nc, Kc, block_k, block_n)
  auto w_k = context.at_weight_.dim() == 2
      ? context.at_weight_.size(1)
      : context.at_weight_.size(1) * context.at_weight_.size(2);
  TORCH_CHECK(
      input.size(input.dim() - 1) == w_k,
      "WOQ linear: input and weight shapes do not match, got k = ",
      input.size(input.dim() - 1),
      " and ",
      w_k,
      " respectively.");
  auto input_ = input.contiguous();
  return woq_linear_eltwise_kernel(
      input_,
      context.at_weight_,
      context.scales_list_,
      context.zero_points_list_,
      context.bias_list_,
      post_op,
      scalars,
      algorithm,
      context.is_int4_,
      context.lowp_mode_,
      context.num_concats_);
}

// Registered as JIT op
at::Tensor woq_linear_eltwise_run(
    const at::Tensor& input,
    const at::Tensor& op_context,
    const c10::string_view& post_op,
    const torch::List<c10::optional<at::Scalar>>& scalars,
    const c10::optional<c10::string_view>& algorithm) {
  static std::map<c10::string_view, std::string> postop_to_record_name_map = {
      {"relu", "torch_ipex::woq_linear_relu_run"},
      {"gelu", "torch_ipex::woq_linear_gelu_run"},
  };
  RECORD_FUNCTION(
      postop_to_record_name_map[post_op], c10::ArrayRef<c10::IValue>({}));
  return reinterpret_cast<IpexWoqLinearOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run_eltwise(input, post_op, scalars, algorithm);
}

// Called by IpexWoqLinearOpContext::run_add
at::Tensor run_add(
    ContextLinearWoq& context,
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha) {
  // TPP kernel packs weight to 4d (Nc, Kc, block_k, block_n)
  auto w_k = context.at_weight_.dim() == 2
      ? context.at_weight_.size(1)
      : context.at_weight_.size(1) * context.at_weight_.size(2);
  TORCH_CHECK(
      input.size(input.dim() - 1) == w_k,
      "WOQ linear: input and weight shapes do not match, got k = ",
      input.size(input.dim() - 1),
      " and ",
      w_k,
      " respectively.");
  auto input_ = input.contiguous();
  return woq_linear_add_kernel(
      input_,
      context.at_weight_,
      context.scales_list_,
      context.zero_points_list_,
      context.bias_list_,
      context.is_int4_,
      context.lowp_mode_,
      context.num_concats_,
      accumu,
      alpha);
}

// Called by IpexWoqLinearOpContext::run_add_relu
at::Tensor run_add_relu(
    ContextLinearWoq& context,
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha) {
  // TPP kernel packs weight to 4d (Nc, Kc, block_k, block_n)
  auto w_k = context.at_weight_.dim() == 2
      ? context.at_weight_.size(1)
      : context.at_weight_.size(1) * context.at_weight_.size(2);
  TORCH_CHECK(
      input.size(input.dim() - 1) == w_k,
      "WOQ linear: input and weight shapes do not match, got k = ",
      input.size(input.dim() - 1),
      " and ",
      w_k,
      " respectively.");
  auto input_ = input.contiguous();
  auto output = woq_linear_kernel(
      input_,
      context.at_weight_,
      context.scales_list_,
      context.zero_points_list_,
      context.bias_list_,
      context.is_int4_,
      context.lowp_mode_,
      context.num_concats_);
  at::add_out(accumu, output, accumu, alpha.value());
  at::relu_(accumu);
  return accumu;
}

// Called by IpexWoqLinearOpContext::run_add
at::Tensor run_add(
    ContextLinearWoq& context,
    const at::Tensor& input,
    const std::vector<at::Tensor>& others) {
  // TPP kernel packs weight to 4d (Nc, Kc, block_k, block_n)
  auto w_k = context.at_weight_.dim() == 2
      ? context.at_weight_.size(1)
      : context.at_weight_.size(1) * context.at_weight_.size(2);
  TORCH_CHECK(
      input.size(input.dim() - 1) == w_k,
      "WOQ linear: input and weight shapes do not match, got k = ",
      input.size(input.dim() - 1),
      " and ",
      w_k,
      " respectively.");
  auto input_ = input.contiguous();
  return woq_linear_add_kernel(
      input_,
      context.at_weight_,
      context.scales_list_,
      context.zero_points_list_,
      context.bias_list_,
      context.is_int4_,
      context.lowp_mode_,
      context.num_concats_,
      others);
}

// Called by IpexWoqLinearOpContext::run_add_add
at::Tensor run_add_add(
    ContextLinearWoq& context,
    const at::Tensor& input,
    const std::vector<at::Tensor>& others) {
  // TPP kernel packs weight to 4d (Nc, Kc, block_k, block_n)
  auto w_k = context.at_weight_.dim() == 2
      ? context.at_weight_.size(1)
      : context.at_weight_.size(1) * context.at_weight_.size(2);
  TORCH_CHECK(
      input.size(input.dim() - 1) == w_k,
      "WOQ linear: input and weight shapes do not match, got k = ",
      input.size(input.dim() - 1),
      " and ",
      w_k,
      " respectively.");
  auto input_ = input.contiguous();
  return woq_linear_add_add_kernel(
      input_,
      context.at_weight_,
      context.scales_list_,
      context.zero_points_list_,
      context.bias_list_,
      context.is_int4_,
      context.lowp_mode_,
      context.num_concats_,
      others);
}

// Registered as JIT op
at::Tensor woq_linear_add_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const at::Tensor& op_context) {
  RECORD_FUNCTION(
      "torch_ipex::woq_linear_add_run", c10::ArrayRef<c10::IValue>({}));
  return reinterpret_cast<IpexWoqLinearOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run_add(input, accumu, alpha);
}

// Registered as JIT op
at::Tensor woq_linear_add_relu_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const at::Tensor& op_context) {
  RECORD_FUNCTION(
      "torch_ipex::woq_linear_add_relu_run", c10::ArrayRef<c10::IValue>({}));
  return reinterpret_cast<IpexWoqLinearOpContext*>(
             op_context.data_ptr<int64_t>()[0])
      ->run_add_relu(input, accumu, alpha);
}

at::Tensor pack(ContextLinearWoq& context, const at::Tensor& tensor) {
  return tensor;
}

at::Tensor unpack(ContextLinearWoq& context, const at::Tensor& tensor) {
  // By using different kernels, the packed weight dim can be 2 or 4
  // Return result directly if dim == 2
  // For dim == 4, make a new quantized tensor and return.
  // For padded weight (int4), make a slice of it.
  auto unpacked_weight =
      woq_linear_unpack_weight(tensor, context.is_int4_, context.lowp_mode_);
  if (tensor.dim() > 2) {
    auto scales = context.scales_list_[0];
    auto zero_points = context.zero_points_list_[0];
    if (context.is_int4_) {
      auto unpacked_shape = unpacked_weight.sizes().vec(); // = N * K/2
      auto shape = context.orig_wei_shape_.has_value()
          ? context.orig_wei_shape_.value()
          : std::vector<int64_t>({unpacked_shape[0], unpacked_shape[1] * 2});
      at::Tensor qweight = at::_empty_per_channel_affine_quantized(
          shape,
          scales,
          zero_points,
          0,
          device(c10::kCPU).dtype(c10::kQUInt4x2));
      assert(qweight.numel() % 2 == 0);
      std::memcpy(
          qweight.data_ptr(), unpacked_weight.data_ptr(), qweight.numel() / 2);
      return qweight;
    } else { // int8
      return at::_make_per_channel_quantized_tensor(
          unpacked_weight.int_repr(), scales, zero_points.to(c10::kInt), 0);
    }
  }
  return unpacked_weight;
}

} // namespace woq_linear
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
#endif