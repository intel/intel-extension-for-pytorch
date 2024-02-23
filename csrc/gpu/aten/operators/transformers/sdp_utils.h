#pragma once
#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/SymFloat.h>
#include <runtime/Device.h>
#include "../xetla/mha.h"
#include "utils/LogUtils.h"

using namespace at;
using namespace gpu::xetla;
namespace sdp {

// This helper function creates a constexpr std::array
// From a compile time list of values
template <typename V, typename... T>
inline constexpr auto array_of(T&&... t) -> std::array<V, sizeof...(T)> {
  return {{std::forward<T>(t)...}};
}

// The same definition as PyTorch
// We define here because head file in PyTorch is not exposed
enum class SDPBackend {
  error = -1,
  math = 0,
  flash_attention = 1,
  efficient_attention = 2
};

struct sdp_params {
  const at::Tensor& query;
  const at::Tensor& key;
  const at::Tensor& value;
  const c10::optional<at::Tensor> attn_mask;
  double dropout;
  bool is_causal;
};

inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    c10::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : c10::SymFloat(query.sym_size(-1)).sqrt();
  return c10::SymFloat(softmax_scale);
}

inline c10::SymFloat native_calculate_scale(
    const at::Tensor& query,
    c10::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}

inline c10::optional<at::Tensor> convert_boolean_attn_mask(
    const c10::optional<at::Tensor>& attn_mask,
    caffe2::TypeMeta dtype) {
  // Pass through
  if (!attn_mask.has_value()) {
    return c10::nullopt;
  }
  // Convert boolean mask to additive mask; need to invert mask to indicate what
  // to mask *out*.
  if (attn_mask->dtype() == at::kBool) {
    auto new_attn_mask = at::zeros_like(attn_mask.value(), dtype);
    // TODO Use the max type of the input and output
    new_attn_mask.masked_fill_(
        attn_mask->logical_not(), -std::numeric_limits<double>::infinity());
    return new_attn_mask;
  }
  // Otherwise, attn_mask represents an additive attention tensor
  return attn_mask;
}

inline XetlaType aten_to_Xetla_dtype(const Tensor& input) {
  XetlaType xeType;
  if (input.scalar_type() == kHalf) {
    xeType = XetlaType::fp16;
  } else if (input.scalar_type() == kBFloat16) {
    xeType = XetlaType::bf16;
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "XPU scaled_dot_product_efficient_attention only supports half and bfloat16");
  }
  return xeType;
}

inline bool xetla_supported(sdp::sdp_params params) {
  bool is_supported = false;
#if defined(USE_XETLA)
  if (dpcppGetDeviceHasXMX()) {
    DeviceId curDevID;
    AT_DPCPP_CHECK(dpcppGetDevice(&curDevID));
    if ((params.query.dtype() == at::kHalf ||
         params.query.dtype() == at::kBFloat16) &&
        Settings::I().has_2d_block_array(curDevID)) {
      if ((params.query.sym_size(-1) * params.query.itemsize() % 128 == 0) &&
          (params.value.sym_size(-1) * params.value.itemsize() % 128 == 0))
        is_supported = true;
    }
  }
#endif
  if (!is_supported) {
    IPEX_DEBUG_LOG(
        "OPS", "", "Your IPEX version currently doesn't support xetla.");
  }
  return is_supported;
}

inline bool input_requires_grad(sdp_params params) {
  const bool any_inputs_require_grad = params.query.requires_grad() ||
      params.key.requires_grad() || params.value.requires_grad();
  const bool gradmode_enabled = at::GradMode::is_enabled();
  return any_inputs_require_grad && gradmode_enabled;
}

inline bool has_for_nested_inputs(sdp_params params) {
  return (
      params.query.is_nested() || params.key.is_nested() ||
      params.value.is_nested());
}

inline bool check_requires_grad_and_nested(sdp_params params) {
  // If we fail both checks then we return false
  if (has_for_nested_inputs(params) && input_requires_grad(params)) {
    IPEX_DEBUG_LOG(
        "OPS",
        "",
        "Memory efficient attention currently doesn't support training with NT inputs.");
    return false;
  }
  return true;
}

inline bool check_tensor_shapes(sdp_params params) {
  auto query_dim = params.query.dim();
  if (!(query_dim == params.key.dim() && query_dim == params.value.dim() &&
        (query_dim == 4))) {
    IPEX_DEBUG_LOG(
        "OPS",
        "",
        "Both fused kernels requires query, key and value to be 4 dimensional.");
    return false;
  }
  return true;
}

inline bool try_broadcast_param_size(
    const c10::SymInt q_size,
    const c10::SymInt k_size,
    const c10::SymInt v_size,
    std::string param_name) {
  auto max_size = std::max({q_size, k_size, v_size});
  if ((q_size != max_size && q_size != 1) ||
      (k_size != max_size && k_size != 1) ||
      (v_size != max_size && v_size != 1)) {
    IPEX_DEBUG_LOG(
        "OPS",
        "",
        "Both fused kernels require query, key and value to have broadcastable {}.",
        param_name);
    return false;
  }
  return true;
}

inline bool check_safe_kv_broadcast(at::Tensor param) {
  const auto nt_tensor_impl = at::native::get_nested_tensor_impl(param);
  auto seq_len = nt_tensor_impl->opt_size(2);
  if (!seq_len.has_value()) {
    IPEX_DEBUG_LOG(
        "OPS",
        "",
        "For both fused kernels, if one of key/value batch_size requires "
        "broadcasting and the other does not, then the other must "
        "have a consistent seq_len dim.");
    return false;
  }
  return true;
}

inline bool check_batch_size_and_num_heads(sdp_params params) {
  // This is expected to be called after check_tensor_shapes ensuring that the
  // size() calls won't error since the inputs are all 4 dimensional
  auto q_batch_size = params.query.sym_size(0);
  auto k_batch_size = params.key.sym_size(0);
  auto v_batch_size = params.value.sym_size(0);

  bool has_nested_input = has_for_nested_inputs(params);
  bool same_batch_size =
      q_batch_size == k_batch_size && q_batch_size == v_batch_size;

  // num_heads logic for nested input is checked in
  // check_for_seq_len_0_nested_tensor as there is handling there to make sure
  // num_heads is not ragged
  if (has_nested_input) {
    bool broadcastable_batch_size = true;
    if (!same_batch_size) {
      // try to broadcast batchsize
      broadcastable_batch_size = try_broadcast_param_size(
          q_batch_size, k_batch_size, v_batch_size, "batch size ");

      // if only one of k or v require broadcasting of batch size, the other
      // must have a consistent seq_len dim
      if (broadcastable_batch_size) {
        if (k_batch_size == 1 && v_batch_size != 1 &&
            !check_safe_kv_broadcast(params.value)) {
          return false;
        }
        if (v_batch_size == 1 && k_batch_size != 1 &&
            !check_safe_kv_broadcast(params.key)) {
          return false;
        }
      }
    }
    return broadcastable_batch_size;
  }

  auto q_num_heads = params.query.sym_size(1);
  auto k_num_heads = params.key.sym_size(1);
  auto v_num_heads = params.value.sym_size(1);
  bool same_num_heads =
      q_num_heads == k_num_heads && q_num_heads == v_num_heads;

  if (!(same_batch_size && same_num_heads)) {
    IPEX_DEBUG_LOG(
        "OPS",
        "",
        "For dense inputs, both fused kernels require query, key and value "
        "to have the same batch_size and num_heads. "
        "To broadcast dense inputs, try using unsqueeze and "
        "expand_to before passing them into the kernel.");
    return false;
  }
  return true;
}

inline bool check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
    at::Tensor param,
    std::string param_name) {
  const auto nt_tensor_impl = at::native::get_nested_tensor_impl(param);
  const at::Tensor& sizes = nt_tensor_impl->get_nested_sizes();
  auto num_head_dims = nt_tensor_impl->opt_size(1);
  if (!num_head_dims.has_value()) {
    // num_head_dims is ragged
    IPEX_DEBUG_LOG(
        "OPS",
        "",
        "Fused kernels do not support ragged num_head_dims, {}"
        "has a ragged num_heads.",
        param_name);
    return false;
  }

  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const int64_t n_tensors = param.size(0);
  const int64_t size_tensor_stride = sizes.stride(0);

  // This is being called inside sdp with shape [batch, heads, {seq_len}, dim]
  for (const auto i : c10::irange(n_tensors)) {
    if (sizes_ptr[(i * size_tensor_stride) + 1] == 0) {
      IPEX_DEBUG_LOG(
          "OPS",
          "",
          "Fused kernels do not support seq_len == 0, {}"
          "has a seq len of 0.",
          param_name);
      return false;
    }
  }
  return true;
}

inline bool check_for_seq_len_0_nested_tensor(sdp_params params) {
  // When this function is called we are assured that the nt is dim==4
  if (!has_for_nested_inputs(params)) {
    return true;
  }

  bool q_is_safe = params.query.is_nested()
      ? check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
            params.query, "query ")
      : true;
  // short circuit if any is unsafe
  if (!q_is_safe) {
    return false;
  }

  bool k_is_safe = params.key.is_nested()
      ? check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
            params.key, "key ")
      : true;
  if (!k_is_safe) {
    return false;
  }

  bool v_is_safe = params.value.is_nested()
      ? check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
            params.value, "value ")
      : true;
  if (!v_is_safe) {
    return false;
  }

  // We now know none of the inputs have ragged num_heads, so we can safely
  // access .size(1)
  auto q_num_heads = params.query.size(1);
  auto k_num_heads = params.key.size(1);
  auto v_num_heads = params.value.size(1);
  bool same_num_heads =
      q_num_heads == k_num_heads && q_num_heads == v_num_heads;

  if (!same_num_heads) {
    return try_broadcast_param_size(
        q_num_heads, k_num_heads, v_num_heads, "num heads ");
  }

  return true;
}

inline bool check_nonzero_sequence_lengths(sdp_params params) {
  if (has_for_nested_inputs(params)) {
    // Currently we do not support any masking with NestedTensors
    // This is checked in validate_sdpa_input so this filter func
    // Should have no actually bearing on the kernel selection
    return true;
  }
  // In some cases people will pass in 0 sized tensors, this will
  // cause the fused path to error with unaligned mask
  bool zero_seq_len_q = params.query.sym_size(-2) == 0;
  bool zero_seq_len_k = params.key.sym_size(-2) == 0;
  if (zero_seq_len_q || zero_seq_len_k) {
    IPEX_DEBUG_LOG(
        "OPS",
        "",
        "Both fused kernels do not support zero seq_len_q or seq_len_kv.");
    return false;
  }
  return true;
}

inline bool check_last_dim_stride_equals_1(sdp_params params) {
  if (has_for_nested_inputs(params)) {
    // The stride checking for NestedTensors is done within the kernel
    // And .contiguous will be called if needed
    return true;
  }
  // This function checks that the last dimension of the inputs to
  // fused_attention have stride 1
  bool qkv_strides_equal_1 = params.query.sym_stride(-1) == 1 &&
      params.key.sym_stride(-1) == 1 && params.value.sym_stride(-1) == 1;
  bool mask_stride_equal_1 = params.attn_mask.has_value()
      ? params.attn_mask.value().sym_stride(-1) == 1
      : true;
  if (!(qkv_strides_equal_1 && mask_stride_equal_1)) {
    IPEX_DEBUG_LOG(
        "OPS",
        "",
        "Both fused kernels require the last dimension of the input to have stride 1. ");
    return false;
  }
  return true;
}

inline bool use_mem_efficient_attention(sdp::sdp_params params) {
  //  Define gate functions that determine if a flash kernel can be ran
  constexpr auto constraints = sdp::array_of<bool (*)(sdp::sdp_params)>(
      sdp::xetla_supported,
      sdp::check_requires_grad_and_nested,
      sdp::check_tensor_shapes,
      sdp::check_batch_size_and_num_heads,
      sdp::check_for_seq_len_0_nested_tensor,
      sdp::check_nonzero_sequence_lengths,
      sdp::check_last_dim_stride_equals_1);
  for (auto& constraint : constraints) {
    if (!constraint(params)) {
      return false;
    }
  }
  return true;
}

template <int alignment>
Tensor pad_bias(const Tensor& attn_bias) {
  auto last_dim_size = attn_bias.sym_size(-1);
  auto pad_count = alignment - (last_dim_size % alignment);
  auto padded_bias = at::pad_symint(attn_bias, {c10::SymInt(0), pad_count});
  return padded_bias.slice_symint(-1, 0, last_dim_size);
}

} // namespace sdp