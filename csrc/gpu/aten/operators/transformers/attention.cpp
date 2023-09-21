#include <ATen/ATen.h>
#include <utils/DPCPP.h>

//#include <xetla/kernels/SDP/mha_forward.h>
#include <ATen/record_function.h>
#include "../comm/ATDispatch.h"
#include "../xetla/mha.h"
#include "NaiveScaledDotProduct.h"
#include "sdp_utils.h"
#include "sdp_utils_cpp.h"
#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {

std::tuple<Tensor, Tensor> _scaled_dot_product_efficient_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool compute_log_sumexp,
    bool is_causal) {
#if defined(USE_XETLA)
  auto output = at::empty_like(query);
  auto output_lm = at::empty_like(query);
  auto dpcpp_queue = dpcppGetCurrentQueue();
  bool is_strided = (key.strides()[1] == key.sizes()[3] &&
                     key.strides()[2] == (key.sizes()[1] * key.sizes()[3]))
      ? true
      : false;
  if (is_causal) {
    if (!is_strided) {
      RECORD_FUNCTION("xetla_fsdp_forward_no_mask_no_strided", {});
      gpu::xetla::fmha_forward_op_causal(
          dpcpp_queue,
          query.data_ptr(),
          key.data_ptr(),
          value.data_ptr(),
          output.data_ptr(),
          query.size(0),
          query.size(1),
          query.size(3),
          query.size(2),
          key.size(2));
    } else {
      RECORD_FUNCTION("xetla_fsdp_forward_no_mask", {});
      gpu::xetla::fmha_forward_kernel(
          dpcpp_queue,
          query.data_ptr(),
          key.data_ptr(),
          value.data_ptr(),
          /* alibi */ nullptr,
          /* attn_mask */ nullptr,
          /* dropout_mask */ nullptr,
          output.data_ptr(),
          /* alpha */ sycl::rsqrt(float(query.size(3))),
          /* beta */ 1.0f,
          /* dropout_p */ 1.0f,
          query.size(0),
          query.size(1),
          query.size(3),
          query.size(2),
          key.size(2),
          /* ablibi padded size */ 0,
          /* attn_mask padded size */ 0,
          is_causal,
          false);
    }
  } else {
    if (!is_strided) {
      RECORD_FUNCTION("xetla_fsdp_forward_no_mask_no_causal_no_strided", {});
      gpu::xetla::fmha_forward_op(
          dpcpp_queue,
          query.data_ptr(),
          key.data_ptr(),
          value.data_ptr(),
          output.data_ptr(),
          query.size(0),
          query.size(1),
          query.size(3),
          query.size(2),
          key.size(2));
    } else {
      RECORD_FUNCTION("xetla_fsdp_forward_no_mask_no_causal", {});
      gpu::xetla::fmha_forward_op_strided(
          dpcpp_queue,
          query.data_ptr(),
          key.data_ptr(),
          value.data_ptr(),
          output.data_ptr(),
          query.size(0),
          query.size(1),
          query.size(3),
          query.size(2),
          key.size(2));
    }
  }
  return std::forward_as_tuple(output, output_lm);
#else
  auto result = naive_scaled_dot_product(query, key, value, is_causal);
  return std::forward_as_tuple(std::get<0>(result), std::get<1>(result));
#endif
}

std::tuple<
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    Tensor>
_scaled_dot_product_flash_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask) {
  TORCH_CHECK(
      false,
      "'_scaled_dot_product_flash_attention' hasn't been implemented, we should have falled back to the math path.");
  // TODO: Implement flash attention algorithm.
}

int64_t _fused_sdp_choice(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal) {
  sdp::sdp_params kernel_params{
      query_, key, value, attn_mask_.has_value(), dropout_p, is_causal};
  auto backend = select_sdp_backend(kernel_params);
  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(
        false,
        "No viable backend for scaled_dot_product_attention was found. ",
        "This is likely due to turning off both the math kernel and the fused kernels.");
  }
  return static_cast<int64_t>(backend);
}

inline void validate_sdpa_input(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal) {
  TORCH_CHECK(
      query_.dtype() == key.dtype() && query_.dtype() == value.dtype(),
      "Expected query, key, and value to have the same dtype, but got query.dtype: ",
      query_.dtype(),
      " key.dtype: ",
      key.dtype(),
      " and value.dtype: ",
      value.dtype(),
      " instead.");
  TORCH_CHECK(
      query_.device() == key.device() && query_.device() == value.device(),
      "Expected query, key, and value to have the same device type, but got query.device: ",
      query_.device(),
      " key.device: ",
      key.device(),
      " and value.device: ",
      value.device(),
      " instead.");
  TORCH_CHECK(
      query_.dim() >= 2 && key.dim() >= 2 && value.dim() >= 2,
      "Expected query, key, and value to all be  at least 2 dimensional, but got query.dim: ",
      query_.dim(),
      " key.dim: ",
      key.dim(),
      " and value.dim: ",
      value.dim(),
      " instead.");
  if (attn_mask_.has_value()) {
    auto mask_dtype = attn_mask_->dtype();
    TORCH_CHECK(
        mask_dtype == at::kBool || mask_dtype == query_.dtype(),
        "Expected attn_mask dtype to be bool or to match query dtype, but got attn_mask.dtype: ",
        mask_dtype,
        " and  query.dtype: ",
        query_.dtype(),
        " instead.");
  }
  return;
}

Tensor scaled_dot_product_attention(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal) {
  validate_sdpa_input(query_, key, value, attn_mask_, dropout_p, is_causal);
  int64_t choice_int = static_cast<int64_t>(sdp::SDPBackend::math);
  choice_int = at::_fused_sdp_choice(
      query_, key, value, attn_mask_, dropout_p, is_causal);
  sdp::SDPBackend backend = static_cast<sdp::SDPBackend>(choice_int);
  switch (backend) {
    case sdp::SDPBackend::flash_attention: {
      TORCH_WARN(
          "flash_attention algorithm hasn't been implemented, we will fall back to the math path.")
      return std::get<0>(at::_scaled_dot_product_attention_math(
          query_, key, value, attn_mask_, dropout_p, is_causal));
    }
    case sdp::SDPBackend::efficient_attention: {
      bool compute_logsumexp =
          (query_.requires_grad() || key.requires_grad() ||
           value.requires_grad());
      auto out_and_lse = at::_scaled_dot_product_efficient_attention(
          query_, key, value, compute_logsumexp, is_causal);
      return std::get<0>(out_and_lse);
    }
    case sdp::SDPBackend::math:
      return std::get<0>(at::_scaled_dot_product_attention_math(
          query_, key, value, attn_mask_, dropout_p, is_causal));
    default:
      TORCH_CHECK(
          false,
          "No viable backend for scaled_dot_product_attention was found.");
      return Tensor();
  }
}

Tensor xetla_fsdp_forward_atten_mask_alibi_strided(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& alibi,
    const c10::optional<Tensor>& attn_mask,
    const c10::optional<Tensor>& head_mask,
    const double alpha,
    const double beta,
    const double dropout_p,
    bool is_causal,
    bool seq_last) {
  TORCH_CHECK(
      !head_mask.has_value(),
      "Unsupported feature in fsdp kernel, head_mask ...");
  auto output = at::empty_like(query);
  auto dpcpp_queue = dpcppGetCurrentQueue();
  RECORD_FUNCTION("xetla_fsdp_forward_atten_mask_alibi_strided", {});

  // check alibi padded
  uint32_t alibi_padded_block_size = 0;
  if (alibi.has_value()) {
    alibi_padded_block_size = alibi.value().size(-1);
    TORCH_CHECK(
        (alibi_padded_block_size * key.itemsize() % 8 == 0),
        "XeTLA SDP Alibi needs 8bytes aligned on leading dimension ...");
  }

  // check attn_mask padded
  uint32_t attn_mask_padded_block_size = 0;
  if (attn_mask.has_value()) {
    attn_mask_padded_block_size = attn_mask.value().size(-1);
    TORCH_CHECK(
        (attn_mask_padded_block_size * key.itemsize() % 8 == 0),
        "XeTLA SDP Attention mask needs 8bytes aligned on leading dimension ...");
  }

  gpu::xetla::fmha_forward_kernel(
      dpcpp_queue,
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      alibi.has_value() ? alibi.value().data_ptr() : (void*)nullptr,
      attn_mask.has_value() ? attn_mask.value().data_ptr() : (void*)nullptr,
      nullptr,
      output.data_ptr(),
      alpha,
      beta,
      dropout_p,
      query.size(0),
      query.size(1),
      query.size(3),
      query.size(2),
      key.size(2),
      alibi_padded_block_size,
      attn_mask_padded_block_size,
      is_causal,
      seq_last);
  return output;
}

// @brief
// *query       shape  : [bs * beam, num_head, q_seq_len, head_dim]
//              layout : [q_seq_len, bs * beam, num_head, head_dim]
// *key         shape  : [bs, num_head, kv_in_len, head_dim]
//              layout : [kv_in_len, bs, num_head, head_dim]
// *value       shape  : [bs, num_head, kv_in_len, head_dim]
//              layout : [kv_in_len, bs, num_head, head_dim]
// *key_cache   shape  : [bs * beam, num_head, kv_out_len, head_dim]
//              layout : [kv_out_len, bs * beam, num_head, head_dim]
// *value_cache shape  : [bs * beam, num_head, kv_out_len, head_dim]
//              layout : [kv_out_len, bs * beam, num_head, head_dim]
// *index       shape  : [kv_out_len, bs * beam]
//              layout : [kv_out_len, bs * beam]
// *output      shape  : [bs * beam, num_head, kv_in_len + kv_out_len, head_dim]
//              layout : [bs * beam, kv_in_len + kv_out_len, num_head, head_dim]
// *timestep           : current time step of output seq
Tensor xetla_fsdp_index_forward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& key_cache,
    const Tensor& value_cache,
    const Tensor& index,
    const c10::optional<Tensor>& alibi,
    const c10::optional<Tensor>& attn_mask,
    const c10::optional<Tensor>& head_mask,
    const int64_t timestep,
    const double alpha,
    const double beta,
    const double dropout_p,
    bool is_causal) {
  TORCH_CHECK(
      !head_mask.has_value(),
      "Unsupported feature in fsdp kernel, head_mask ...");

  // check alibi padded
  uint32_t alibi_padding = 0;
  if (alibi.has_value()) {
    alibi_padding = alibi.value().size(-1);
    TORCH_CHECK(
        (alibi_padding * key.itemsize() % 8 == 0),
        "XeTLA SDP Alibi needs 8bytes aligned on leading dimension ...");
  }

  // check attn_mask padded
  uint32_t attn_mask_padding = 0;
  if (attn_mask.has_value()) {
    attn_mask_padding = attn_mask.value().size(-1);
    TORCH_CHECK(
        (attn_mask_padding * key.itemsize() % 8 == 0),
        "XeTLA SDP Attention mask needs 8bytes aligned on leading dimension ...");
  }

  uint32_t beam_width = query.size(0) / key.size(0);
  uint32_t num_keys_in = key.size(2);
  uint32_t num_keys_out = key_cache.size(2);
  auto output = at::empty_like(query);
  auto dpcpp_queue = dpcppGetCurrentQueue();
  RECORD_FUNCTION("xetla_fsdp_index_forward", {});
  gpu::xetla::fmha_forward_index_kernel(
      dpcpp_queue,
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      key_cache.data_ptr(),
      value_cache.data_ptr(),
      index.data_ptr<int32_t>(),
      alibi.has_value() ? alibi.value().data_ptr() : (void*)nullptr,
      attn_mask.has_value() ? attn_mask.value().data_ptr() : (void*)nullptr,
      nullptr, /* dropout */
      output.data_ptr(),
      timestep,
      alpha,
      beta,
      dropout_p,
      key.size(0),
      beam_width,
      query.size(1),
      query.size(3),
      query.size(2),
      num_keys_in,
      num_keys_out,
      alibi_padding,
      attn_mask_padding,
      is_causal);
  return output;
}
} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "xetla_fsdp_forward_atten_mask_alibi_strided.xpu",
      at::AtenIpexTypeXPU::xetla_fsdp_forward_atten_mask_alibi_strided,
      c10::DispatchKey::XPU);
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "xetla_fsdp_index_forward.xpu",
      at::AtenIpexTypeXPU::xetla_fsdp_index_forward,
      c10::DispatchKey::XPU);
}
} // namespace
