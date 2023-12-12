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
    std::vector<int64_t>&& weight_shape,
    at::Tensor&& scales,
    at::Tensor&& zero_points,
    c10::optional<at::Tensor>&& bias,
    c10::optional<int64_t> batch_size,
    bool is_int4,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t num_concats,
    int64_t act_quant_mode) {
  RECORD_FUNCTION(
      "ipex_prepack::createWoqLinearPrePackOpContext",
      c10::ArrayRef<c10::IValue>({}));

  return IpexWoqLinearOpContext::create_context(
      std::move(weight),
      std::move(weight_shape),
      std::move(scales),
      std::move(zero_points),
      std::move(bias),
      batch_size,
      is_int4,
      group_size,
      lowp_mode,
      num_concats,
      act_quant_mode);
}

c10::intrusive_ptr<WoqLinearOpContext> createWoqLinearPrePackOpContextInt4(
    at::Tensor&& weight,
    at::Tensor&& scales,
    at::Tensor&& zero_points,
    c10::optional<at::Tensor>&& bias,
    c10::optional<int64_t> batch_size,
    int64_t group_size, // group_size along input channel
    int64_t lowp_mode,
    int64_t num_concats,
    int64_t act_quant_mode) {
  RECORD_FUNCTION(
      "ipex_prepack::createWoqLinearPrePackOpContextInt4",
      c10::ArrayRef<c10::IValue>({}));
  // clang-format off
  // From
  // Weight dtype = int32 (uint4 * 8) or uint8 (4bit * 2), scale dtype = fp16,
  // zero points dtype = int32 (int4 * 8)
  // To
  // Weight dtype = quint4x2, scale dtype = fp32, zero points dtype = fp32
  // There might be an extra output channel in weight and scales.
  // clang-format on
  auto scales_fp32 = scales.squeeze().to(c10::ScalarType::Float);

  at::Tensor zp_fp32;

  if (zero_points.scalar_type() == c10::kInt) {
    // Two cases: (1) each int32 contains 8 values of zero points
    // (2) each int32 is a single value of zero point
    if (zero_points.numel() != scales_fp32.numel()) {
      // Assume group_size > 0 and zero point data are compressed
      TORCH_CHECK(scales_fp32.dim() == 2 && zero_points.dim() == 2)
      TORCH_CHECK(scales_fp32.size(0) == zero_points.size(0))
      auto num_row = scales_fp32.size(0);
      auto num_col = scales_fp32.size(1);
      auto num_col_zp = zero_points.size(1);
      // Convert compressed zero points to float
      zp_fp32 = at::empty_like(scales_fp32);
      float* zp_fp32_ptr = reinterpret_cast<float*>(zp_fp32.data_ptr());
      uint32_t* zp_int32_ptr =
          reinterpret_cast<uint32_t*>(zero_points.data_ptr());
      for (size_t i = 0; i < num_row; ++i) {
        for (size_t j = 0; j < num_col; ++j) {
          zp_fp32_ptr[i * num_col + j] =
              (float)((zp_int32_ptr[i * num_col_zp + j / 8] >> ((j % 8) * 4)) & 0xf);
        }
      }
    } else if (zero_points.numel() == scales_fp32.numel()) {
      // Not compressed
      zp_fp32 = zero_points.squeeze().to(c10::kFloat);
    } else {
      TORCH_CHECK(false, "IPEX WOQ INT4: unexpected zero points size");
    }
  } else {
    zp_fp32 = zero_points.squeeze().to(c10::kFloat);
  }
  // Support two cases here:
  // 1. fp32/bf16 weight after calibration
  // 2. int4 weight after calibration, quantized and compressed, as int32/uint8
  at::Tensor weight_int4;
  std::vector<int64_t> weight_shape(2);
  if (weight.scalar_type() == c10::kInt || weight.scalar_type() == c10::kByte) {
    // Create empty weight with desired options then copy data
    int64_t N = weight.size(0);
    int64_t K_compressed = weight.size(1);
    int64_t K_uint8 =
        weight.scalar_type() == c10::kInt ? K_compressed * 8 / 2 : K_compressed;
    weight_shape[0] = N;
    weight_shape[1] = K_uint8 * 2;
    std::vector<int64_t> weight_size = {N, K_uint8};
    // Create an empty uint8 weight to hold int4 data
    weight_int4 = at::empty(weight_size, device(c10::kCPU).dtype(c10::kByte));
    auto sizeof_dtype = weight.scalar_type() == c10::kInt
        ? sizeof(uint32_t)
        : sizeof(unsigned char);
    std::memcpy(
        weight_int4.data_ptr(),
        weight.data_ptr(),
        weight.numel() * sizeof_dtype);
  } else if (
      weight.scalar_type() == c10::kBFloat16 ||
      weight.scalar_type() == c10::kFloat ||
      weight.scalar_type() == c10::kHalf) {
    weight_shape[0] = weight.size(0);
    weight_shape[1] = weight.size(1);
    auto weight_fp32 = weight.to(c10::kFloat);
    at::Tensor weight_int4_as_uint8;
    if (group_size > 0) {
      auto weight_view =
          weight_fp32.view({-1, weight.size(1) / group_size, group_size});
      auto scale_view = scales_fp32.unsqueeze(2);
      auto zp_view = zp_fp32.unsqueeze(2);
      weight_int4_as_uint8 =
          at::round(weight_view / scale_view + zp_view).to(c10::kByte);
    } else {
      auto scale_view = scales_fp32.unsqueeze(1);
      auto zp_view = zp_fp32.unsqueeze(1);
      weight_int4_as_uint8 =
          at::round(weight / scale_view + zp_view).to(c10::kByte);
    }
    weight_int4_as_uint8 = weight_int4_as_uint8.view(weight_shape);
    using at::indexing::None;
    using at::indexing::Slice;
    at::Tensor even_columns =
        weight_int4_as_uint8.index({Slice(), Slice(1, None, 2)});
    even_columns = even_columns.bitwise_left_shift(4);
    at::Tensor odd_columns =
        weight_int4_as_uint8.index({Slice(), Slice(None, None, 2)});
    weight_int4 = even_columns.bitwise_or(odd_columns);
  } else {
    TORCH_CHECK(
        false,
        "IPEX WOQ INT4: unexpected weight data type: ",
        weight.scalar_type());
  }
  return IpexWoqLinearOpContext::create_context(
      std::move(weight_int4),
      std::move(weight_shape),
      std::move(scales_fp32),
      std::move(zp_fp32),
      std::move(bias),
      batch_size,
      /*is_int4*/ true,
      group_size,
      lowp_mode,
      num_concats,
      act_quant_mode);
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
    std::vector<int64_t>& weight_shape,
    at::Tensor& scales,
    at::Tensor& zero_points,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<int64_t> batch_size,
    bool is_int4,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t num_concats,
    int64_t act_quant_mode) {
  auto packed_weight = woq_linear_pack_weight(
      weight, weight_shape, is_int4, group_size, lowp_mode);
  auto packed_shape = packed_weight.sizes();
  int64_t N = weight.size(0);
  int64_t K = weight.size(1);
  // If OC is not a multiple of BLOCK_N, it may be padded.
  bool oc_is_padded = (packed_shape.size() == 4 && is_int4 &&
                       packed_shape[0] * packed_shape[3] * 2 != N) ||
      (packed_shape.size() == 4 && !is_int4 &&
       packed_shape[0] * packed_shape[3] != N) ||
      (packed_shape.size() == 2 && packed_shape[0] != N);
  auto zero_points_float = zero_points.to(c10::kFloat);
  if (oc_is_padded) {
    int64_t padded_N = packed_shape.size() == 4
        ? (is_int4 ? packed_shape[0] * packed_shape[3] * 2
                   : packed_shape[0] * packed_shape[3])
        : packed_shape[0];
    std::vector<int64_t> pad_vec = scales.dim() == 1
        ? std::vector<int64_t>({0, padded_N - N})
        : std::vector<int64_t>({0, 0, 0, padded_N - N});
    auto scales_padded = at::pad(scales, pad_vec, "constant", 1.f);
    auto zero_points_padded =
        at::pad(zero_points_float, pad_vec, "constant", 0.f);
    if (bias.has_value()) {
      auto bias_padded =
          at::pad(bias.value(), {0, padded_N - N}, "constant", 0.f);
      return ContextLinearWoq(
          std::move(packed_weight),
          std::move(weight_shape),
          std::move(scales_padded),
          std::move(zero_points_padded),
          c10::make_optional(bias_padded),
          is_int4,
          group_size,
          lowp_mode,
          num_concats,
          act_quant_mode);
    } else {
      return ContextLinearWoq(
          std::move(packed_weight),
          std::move(weight_shape),
          std::move(scales_padded),
          std::move(zero_points_padded),
          c10::nullopt,
          is_int4,
          group_size,
          lowp_mode,
          num_concats,
          act_quant_mode);
    }
  }
  return ContextLinearWoq(
      std::move(packed_weight),
      std::move(weight_shape),
      std::move(scales),
      std::move(zero_points_float),
      bias.has_value() ? c10::make_optional(*bias) : c10::nullopt,
      is_int4,
      group_size,
      lowp_mode,
      num_concats,
      act_quant_mode);
}

at::Tensor run(ContextLinearWoq& context, const at::Tensor& input) {
  // TPP kernel packs weight to 4d (Nc, Kc, block_k, block_n)
  auto w_k = context.weight_shape_[1];
  TORCH_CHECK(
      input.size(input.dim() - 1) == w_k,
      "WOQ linear: input and weight shapes do not match, got k = ",
      input.size(input.dim() - 1),
      " and ",
      w_k,
      " respectively.");
  auto input_ = input.contiguous();
  if (context.weight_shape_[0] != context.at_weight_.size(0)) {
    auto res = woq_linear_kernel(
        input_,
        context.at_weight_,
        context.scales_list_,
        context.zero_points_list_,
        context.bias_list_,
        context.is_int4_,
        context.group_size_,
        context.lowp_mode_,
        context.num_concats_,
        context.act_quant_mode_);
    // weight shape is [N by K], output shape is [M by N] or [batch by M by N]
    int64_t N = context.weight_shape_[0];
    return at::narrow(res, /*dim*/ -1, /*start*/ 0, /*end*/ N);
  }
  return woq_linear_kernel(
      input_,
      context.at_weight_,
      context.scales_list_,
      context.zero_points_list_,
      context.bias_list_,
      context.is_int4_,
      context.group_size_,
      context.lowp_mode_,
      context.num_concats_,
      context.act_quant_mode_);
}

// Called by IpexWoqLinearOpContext::run_eltwise
at::Tensor run_eltwise(
    ContextLinearWoq& context,
    const at::Tensor& input,
    const c10::string_view& post_op,
    const torch::List<c10::optional<at::Scalar>>& scalars,
    const c10::optional<c10::string_view>& algorithm) {
  // TPP kernel packs weight to 4d (Nc, Kc, block_k, block_n)
  auto w_k = context.weight_shape_[1];
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
      context.group_size_,
      context.lowp_mode_,
      context.num_concats_,
      context.act_quant_mode_);
}

// Called by IpexWoqLinearOpContext::run_add
at::Tensor run_add(
    ContextLinearWoq& context,
    const at::Tensor& input,
    const std::vector<at::Tensor>& others) {
  // TPP kernel packs weight to 4d (Nc, Kc, block_k, block_n)
  auto w_k = context.weight_shape_[1];
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
      context.group_size_,
      context.lowp_mode_,
      context.num_concats_,
      others,
      context.act_quant_mode_);
}

// Called by IpexWoqLinearOpContext::run_add_add
at::Tensor run_add_add(
    ContextLinearWoq& context,
    const at::Tensor& input,
    const std::vector<at::Tensor>& others) {
  // TPP kernel packs weight to 4d (Nc, Kc, block_k, block_n)
  auto w_k = context.weight_shape_[1];
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
      context.group_size_,
      context.lowp_mode_,
      context.num_concats_,
      others,
      context.act_quant_mode_);
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
      auto shape = context.weight_shape_;
      shape.back() /= 2;
      at::Tensor qweight =
          at::empty(shape, device(c10::kCPU).dtype(c10::kByte));
      assert(qweight.numel() % 2 == 0);
      std::memcpy(
          qweight.data_ptr(), unpacked_weight.data_ptr(), qweight.numel());
      return qweight;
    } else { // int8
      return unpacked_weight;
    }
  }
  return unpacked_weight;
}

} // namespace woq_linear
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
#endif