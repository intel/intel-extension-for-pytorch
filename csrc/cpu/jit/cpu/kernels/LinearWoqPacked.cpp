#ifdef USE_LIBXSMM
#include "LinearWoqPacked.h"
#include <ideep.hpp>
#include "aten/Linear.h"
#include "aten/WeightPack.h"
#include "csrc/cpu/tpp/woq/tla.h"
#include "ideep/IDeepConversions.h"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace woq_linear {

c10::intrusive_ptr<WoqLinearOpContext> createWoqLinearPrePackOpContext(
    at::Tensor&& weight,
    int64_t weight_dtype,
    std::vector<int64_t>&& weight_shape,
    at::Tensor&& scales,
    c10::optional<at::Tensor>&& zero_points,
    c10::optional<at::Tensor>&& bias,
    c10::optional<at::Tensor>&& g_idx,
    c10::optional<int64_t> batch_size,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t act_quant_mode) {
  RECORD_FUNCTION(
      "ipex_prepack::createWoqLinearPrePackOpContext",
      c10::ArrayRef<c10::IValue>({}));

  return IpexWoqLinearOpContext::create_context(
      std::move(weight),
      weight_dtype,
      std::move(weight_shape),
      std::move(scales),
      std::move(zero_points),
      std::move(bias),
      std::move(g_idx),
      batch_size,
      group_size,
      lowp_mode,
      act_quant_mode);
}

c10::intrusive_ptr<WoqLinearOpContext> createWoqLinearPrePackOpContextInt4(
    at::Tensor&& weight,
    at::Tensor&& scales,
    at::Tensor&& zero_points,
    c10::optional<at::Tensor>&& bias,
    c10::optional<at::Tensor>&& g_idx,
    c10::optional<int64_t> batch_size,
    int64_t group_size, // group_size along input channel
    int64_t lowp_mode,
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
      /*weight_dtype*/ WOQ_DTYPE_INT4,
      std::move(weight_shape),
      std::move(scales_fp32),
      std::move(zp_fp32),
      std::move(bias),
      std::move(g_idx),
      batch_size,
      group_size,
      lowp_mode,
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
    int64_t weight_dtype,
    std::vector<int64_t>& weight_shape,
    at::Tensor& scales,
    c10::optional<at::Tensor>& zero_points,
    c10::optional<at::Tensor>& bias,
    c10::optional<at::Tensor>& g_idx,
    const c10::optional<int64_t> batch_size,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t act_quant_mode) {
  at::Tensor packed_weight;
  int64_t N = weight_shape[0];
  int64_t K = weight_shape[1];
  bool is_4bit =
      (weight_dtype == WOQ_DTYPE_INT4 || weight_dtype == WOQ_DTYPE_NF4);
  // GPTQ with act-order
  // Shuffle weight along ic to make channels contiguous in group
  if (is_4bit && group_size > 0 && g_idx.has_value()) {
    // Shuffle weight along ic to make channels contiguous in group
    auto shuffled_weight = woq_shuffle_tensor_by_group_idx</* is_4bit */ true>(
        weight, weight_shape, g_idx.value(), group_size);
    packed_weight = woq_linear_pack_weight(
        shuffled_weight, weight_dtype, weight_shape, group_size, lowp_mode);
  } else {
    packed_weight = woq_linear_pack_weight(
        weight, weight_dtype, weight_shape, group_size, lowp_mode);
  }
  auto packed_shape = packed_weight.sizes();
  // If OC is not a multiple of BLOCK_N, it may be padded.
  bool oc_is_padded = (packed_shape.size() == 4 && is_4bit &&
                       packed_shape[0] * packed_shape[3] * 2 != N) ||
      (packed_shape.size() == 4 && !is_4bit &&
       packed_shape[0] * packed_shape[3] != N) ||
      (packed_shape.size() == 2 && packed_shape[0] != N);
  if (oc_is_padded) {
    int64_t padded_N = packed_shape.size() == 4
        ? (is_4bit ? packed_shape[0] * packed_shape[3] * 2
                   : packed_shape[0] * packed_shape[3])
        : packed_shape[0];
    std::vector<int64_t> pad_vec = scales.dim() == 1
        ? std::vector<int64_t>({0, padded_N - N})
        : std::vector<int64_t>({0, 0, 0, padded_N - N});
    auto scales_padded = at::pad(scales, pad_vec, "constant", 1.f);
    c10::optional<at::Tensor> zero_points_padded = c10::nullopt;
    if (zero_points.has_value() && zero_points.value().defined()) {
      auto zero_points_float = zero_points.value().to(c10::kFloat);
      auto zero_points_tensor_padded =
          at::pad(zero_points_float, pad_vec, "constant", 0.f);
      zero_points_padded = c10::make_optional(zero_points_tensor_padded);
    }
    c10::optional<at::Tensor> bias_padded = c10::nullopt;
    if (bias.has_value() && bias.value().defined()) {
      auto bias_tensor_padded =
          at::pad(bias.value(), {0, padded_N - N}, "constant", 0.f);
      bias_padded = c10::make_optional(bias_tensor_padded);
    }
    return ContextLinearWoq(
        std::move(packed_weight),
        weight_dtype,
        std::move(weight_shape),
        std::move(scales_padded),
        std::move(zero_points_padded),
        std::move(bias_padded),
        std::move(g_idx),
        group_size,
        lowp_mode,
        act_quant_mode);
  }
  c10::optional<at::Tensor> zero_points_float = c10::nullopt;
  if (zero_points.has_value() && zero_points.value().defined()) {
    zero_points_float = c10::make_optional(zero_points.value().to(c10::kFloat));
  }
  return ContextLinearWoq(
      std::move(packed_weight),
      weight_dtype,
      std::move(weight_shape),
      std::move(scales),
      std::move(zero_points_float),
      std::move(bias),
      std::move(g_idx),
      group_size,
      lowp_mode,
      act_quant_mode);
}

static at::Tensor _shuffle_input_channels_if_needed(
    ContextLinearWoq& context,
    const at::Tensor& input) {
  // GPTQ with act-order
  // Shuffle input channels to align with weight
  if (context.is_4bit_ && context.group_size_ > 0 &&
      context.g_idx_.has_value()) {
    auto& g_idx = context.g_idx_.value();
    auto K = input.size(-1);
    std::vector<int64_t> input_shape = {input.numel() / K, K};
    return woq_shuffle_tensor_by_group_idx(
        input, input_shape, g_idx, context.group_size_);
  }
  return input;
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
  // handle GPTQ with act-order
  input_ = _shuffle_input_channels_if_needed(context, input_);
  auto res = woq_linear_kernel(
      input_,
      context.at_weight_,
      context.weight_dtype_,
      context.scales_list_,
      context.zero_points_list_,
      context.bias_list_,
      context.group_size_,
      context.lowp_mode_,
      context.act_quant_mode_);
  if (res.size(-1) != context.weight_shape_[0]) {
    int64_t N = context.weight_shape_[0];
    return at::narrow(res, /*dim*/ -1, /*start*/ 0, /*end*/ N);
  }
  return res;
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
  // handle GPTQ with act-order
  input_ = _shuffle_input_channels_if_needed(context, input_);
  return woq_linear_eltwise_kernel(
      input_,
      context.at_weight_,
      context.weight_dtype_,
      context.scales_list_,
      context.zero_points_list_,
      context.bias_list_,
      post_op,
      scalars,
      algorithm,
      context.group_size_,
      context.lowp_mode_,
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
  // handle GPTQ with act-order
  input_ = _shuffle_input_channels_if_needed(context, input_);
  return woq_linear_add_kernel(
      input_,
      context.at_weight_,
      context.weight_dtype_,
      context.scales_list_,
      context.zero_points_list_,
      context.bias_list_,
      context.group_size_,
      context.lowp_mode_,
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
  // handle GPTQ with act-order
  input_ = _shuffle_input_channels_if_needed(context, input_);
  return woq_linear_add_add_kernel(
      input_,
      context.at_weight_,
      context.weight_dtype_,
      context.scales_list_,
      context.zero_points_list_,
      context.bias_list_,
      context.group_size_,
      context.lowp_mode_,
      others,
      context.act_quant_mode_);
}

at::Tensor pack(ContextLinearWoq& context, const at::Tensor& tensor) {
  return tensor;
}

at::Tensor unpack(ContextLinearWoq& context, const at::Tensor& tensor) {
  // By using different kernels, the packed weight dim can be 2 or 4
  // Return result directly if dim == 2
  // For dim == 4, weight may be padded.
  // For padded weight (int4), make a slice of it.
  auto unpacked_weight = woq_linear_unpack_weight(
      tensor, context.weight_dtype_, context.lowp_mode_);
  // With g_idx, weight's input channels are shuffled along ic so that
  // those in the same group are contiguous.
  // Here we need to shuffle them to the original order.
  if (context.group_size_ > 0 && context.g_idx_.has_value()) {
    auto group_size = context.group_size_;
    auto& g_idx = context.g_idx_.value();
    unpacked_weight = woq_shuffle_weight_back_by_group_idx(
        unpacked_weight, context.weight_shape_, g_idx, group_size);
  }
  if (tensor.dim() > 2) {
    auto scales = context.scales_list_[0];
    auto zero_points = context.zero_points_list_[0];
    if (context.is_4bit_) {
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

template <typename T, typename Tg, bool is_4bit = false>
at::Tensor woq_shuffle_tensor_by_group_idx_impl(
    const at::Tensor& tensor,
    const std::vector<int64_t>& tensor_shape,
    const at::Tensor& g_idx,
    int64_t group_size) {
  // g_idx shape = [ic]
  // i-th element indicates which group tensor[:][i] belongs to.
  // Shuffle tensor along ic to make channels contiguous in group.
  int64_t N = tensor_shape[0];
  int64_t K = tensor_shape[1];
  auto shuffled_tensor = at::zeros_like(tensor, tensor.dtype());
  auto shuffled_tensor_data = reinterpret_cast<T*>(shuffled_tensor.data_ptr());
  auto tensor_data = reinterpret_cast<T*>(tensor.data_ptr());
  auto num_groups = (K + group_size - 1) / group_size;
  auto g_idx_data = reinterpret_cast<Tg*>(g_idx.data_ptr());
#pragma omp parallel for
  for (int64_t i = 0; i < N; ++i) {
    std::vector<int64_t> counts_per_group(num_groups, 0);
    auto stride = is_4bit ? K / 2 : K;
    auto tensor_row_data = tensor_data + i * stride;
    auto shuffled_row_data = shuffled_tensor_data + i * stride;
    for (int64_t j = 0; j < K; ++j) {
      auto g = g_idx_data[j];
      auto new_idx = g * group_size + counts_per_group[g];
      constexpr bool T_is_int8 =
          std::is_same<T, int8_t>() || std::is_same<T, uint8_t>();
      if constexpr (is_4bit && T_is_int8) {
        uint8_t mask = j % 2 ? 0xF0 : 0x0F;
        size_t rshift = j % 2 ? 4 : 0;
        T data = (tensor_row_data[j / 2] & mask) >> rshift;
        shuffled_row_data[new_idx / 2] =
            shuffled_row_data[new_idx / 2] | (new_idx % 2 ? (data << 4) : data);
      } else {
        T data = tensor_row_data[j];
        shuffled_row_data[new_idx] = data;
      }
      ++counts_per_group[g];
    }
  }
  return shuffled_tensor;
}

/**
 * Shuffle activation or weight tensor along input channel according to group
 * index (g_idx), so that input channels in the same group are contiguous to
 * each other.
 *
 * @param is_4bit The tensor stores 4bit data or not
 * @param tensor The tensor to be shuffled. It must be 2d.
 * @param tensor_shape The original shape of the tensor. It is different from
 * tensor.shape() when dtype is int4 since 2 int4 data are packed as one int8.
 * @param g_idx The g_idx tensor contains group index for each input channel.
 * Its shape is [number of input channels]. Indices should be in [0, number of
 * groups).
 * @param group_size The group size of input channels. Used to determine number
 * of groups.
 * @return The shuffled tensor.
 */
template <bool is_4bit>
at::Tensor woq_shuffle_tensor_by_group_idx(
    const at::Tensor& tensor,
    const std::vector<int64_t>& tensor_shape,
    const at::Tensor& g_idx,
    int64_t group_size) {
  at::Tensor out;
  product_dispatcher<
      std::tuple<at::ScalarType, at::ScalarType>,
      std::tuple<
          enumerate_dispatcher<
              at::ScalarType,
              at::kDouble,
              at::kFloat,
              at::kBFloat16,
              at::kHalf,
              at::kChar,
              at::kByte>,
          enumerate_dispatcher<at::ScalarType, at::kInt, at::kLong>>>::
      call(
          std::make_tuple(tensor.scalar_type(), g_idx.scalar_type()),
          [&](auto dtype_tuple) {
            auto tensor_dtype = std::get<0>(dtype_tuple);
            auto g_idx_dtype = std::get<1>(dtype_tuple);
            using t_cpp_type =
                typename c10::impl::ScalarTypeToCPPType<tensor_dtype>::type;
            using g_cpp_type =
                typename c10::impl::ScalarTypeToCPPType<g_idx_dtype>::type;
            out = woq_shuffle_tensor_by_group_idx_impl<
                t_cpp_type,
                g_cpp_type,
                is_4bit>(tensor, tensor_shape, g_idx, group_size);
          },
          [](auto dtype_tuple) {
            TORCH_CHECK(
                false, "Unsupported tensor data type for WOQ with g_idx");
          });
  return out;
}

template <typename T, typename Tg>
at::Tensor woq_shuffle_weight_back_by_group_idx_impl(
    const at::Tensor& qweight,
    const std::vector<int64_t>& weight_shape,
    const at::Tensor& g_idx,
    int64_t group_size) {
  auto N = weight_shape[0];
  auto K = weight_shape[1];
  auto shuffled_tensor = at::zeros_like(qweight, qweight.dtype());
  auto shuffled_tensor_data = reinterpret_cast<T*>(shuffled_tensor.data_ptr());
  auto tensor_data = reinterpret_cast<T*>(qweight.data_ptr());
  auto num_groups = (K + group_size - 1) / group_size;
  auto g_idx_data = reinterpret_cast<Tg*>(g_idx.data_ptr());
#pragma omp parallel for
  for (int64_t i = 0; i < N; ++i) {
    std::vector<int64_t> counts_per_group(num_groups, 0);
    auto stride = K / 2;
    auto tensor_row_data = tensor_data + i * stride;
    auto shuffled_row_data = shuffled_tensor_data + i * stride;
    for (int64_t j = 0; j < K; ++j) {
      auto g = g_idx_data[j];
      T* data_pos =
          tensor_row_data + g * group_size / 2 + counts_per_group[g] / 2;
      uint8_t mask = counts_per_group[g] % 2 ? 0xF0 : 0x0F;
      size_t rshift = counts_per_group[g] % 2 ? 4 : 0;
      T data = (*data_pos & mask) >> rshift;
      shuffled_row_data[j / 2] =
          shuffled_row_data[j / 2] | (j % 2 ? (data << 4) : data);
      ++counts_per_group[g];
    }
  }
  return shuffled_tensor;
}

/**
 * Shuffle weight tensor along input channel according to group index (g_idx)
 * to its original order. It is used for unpacking weight. Data type is assumed
 * INT4.
 *
 * @param qweight The weight to be shuffled. It must be 2d.
 * @param weight_shape The original shape of the weight. It is different from
 * tensor.shape() since 2 int4 data are packed as one int8.
 * @param g_idx The g_idx tensor contains group index for each input channel.
 * Its shape is [number of input channels]. Indices should be in [0, number of
 * groups).
 * @param group_size The group size of input channels. Used to determine number
 * of groups.
 * @return The shuffled tensor.
 */
at::Tensor woq_shuffle_weight_back_by_group_idx(
    const at::Tensor& qweight,
    const std::vector<int64_t>& weight_shape,
    const at::Tensor& g_idx,
    int64_t group_size) {
  at::Tensor out;
  product_dispatcher<
      std::tuple<at::ScalarType, at::ScalarType>,
      std::tuple<
          enumerate_dispatcher<at::ScalarType, at::kChar, at::kByte>,
          enumerate_dispatcher<at::ScalarType, at::kInt, at::kLong>>>::
      call(
          std::make_tuple(qweight.scalar_type(), g_idx.scalar_type()),
          [&](auto dtype_tuple) {
            auto tensor_dtype = std::get<0>(dtype_tuple);
            auto g_idx_dtype = std::get<1>(dtype_tuple);
            using t_cpp_type =
                typename c10::impl::ScalarTypeToCPPType<tensor_dtype>::type;
            using g_cpp_type =
                typename c10::impl::ScalarTypeToCPPType<g_idx_dtype>::type;
            out = woq_shuffle_weight_back_by_group_idx_impl<
                t_cpp_type,
                g_cpp_type>(qweight, weight_shape, g_idx, group_size);
          },
          [](auto dtype_tuple) {
            TORCH_CHECK(
                false, "Unsupported tensor data type for WOQ with g_idx");
          });
  return out;
}

} // namespace woq_linear
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
#endif