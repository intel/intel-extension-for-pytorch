#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/record_function.h>
#include <runtime/Device.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "../comm/ATDispatch.h"
#include "sdp_utils.h"
#include "utils/CustomOperatorRegistration.h"

#include "../xetla/mha.h"

namespace at {
namespace AtenIpexTypeXPU {

inline Tensor _scaled_dot_product_efficient_attention_impl(
    const Tensor& _query,
    const Tensor& _key,
    const Tensor& _value,
    const c10::optional<Tensor>& attn_mask,
    bool is_causal,
    double dropout_p,
    c10::optional<double> scale) {
#if defined(USE_XETLA)
  TORCH_CHECK(
      dpcppGetDeviceHasXMX(),
      "SDP kernel requires XMX, but the current platform has no XMX ...");
  // check attn_mask padded
  uint32_t attn_mask_padded_block_size = 0;
  if (attn_mask.has_value()) {
    attn_mask_padded_block_size = attn_mask.value().size(-1);
    TORCH_CHECK(
        (attn_mask_padded_block_size * _key.itemsize() % 8 == 0),
        "XeTLA SDP Attention mask needs 8bytes aligned on leading dimension ...");
  }

  // make q, k, v strided
  auto query = _query.transpose(1, 2).contiguous().transpose(1, 2);
  auto key = _key.transpose(1, 2).contiguous().transpose(1, 2);
  auto value = _value.transpose(1, 2).contiguous().transpose(1, 2);

  // create strided output
  // size [bs, num_head, qsize, head_size]
  // layout [bs, qsize, num_head, head_size]
  auto output = at::empty_like(query);
  auto dpcpp_queue = dpcppGetCurrentQueue();

  const double softmax_scale =
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(query.size(-1)));

  gpu::xetla::fmha_forward_kernel(
      dpcpp_queue,
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      /* alibi */ nullptr,
      attn_mask.has_value() ? attn_mask.value().data_ptr() : (void*)nullptr,
      /* dropout_mask */ nullptr,
      output.data_ptr(),
      softmax_scale,
      /* beta */ 1.0f,
      dropout_p,
      query.size(0),
      query.size(1),
      query.size(3),
      query.size(2),
      key.size(2),
      /* ablibi padded size */ 0,
      attn_mask_padded_block_size,
      is_causal,
      false);

  return output;
#else
  AT_ERROR("SDP: xetla library not found in compilation");
  // TODO: sycl kernel impl for efficient_attention
  // auto result = naive_scaled_dot_product(query, key, value, is_causal);
  // return std::forward_as_tuple(std::get<0>(result), std::get<1>(result));
#endif
}

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math_impl(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    const c10::optional<Tensor>& dropout_mask,
    c10::optional<double> scale) {
  if (query_.is_nested() || key.is_nested() || value.is_nested()) {
    TORCH_CHECK(
        query_.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
        "scaled_dot_product_attention: If inputs are nested tensors they must be contiguous");
  }
  auto attn_mask = attn_mask_;
  // Naive, composite implementation defined here.

  // Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for
  // math
  bool is_negative_scaling = scale.has_value() && scale.value() < 0.0;
  const auto scaling_factor =
      sdp::calculate_scale(
          query_, is_negative_scaling ? std::abs(scale.value()) : scale)
          .sqrt();

  const auto query = query_ *
      (is_negative_scaling ? c10::SymFloat(0.0) - scaling_factor
                           : scaling_factor);
  if (is_causal) {
    TORCH_CHECK(
        !attn_mask.has_value(),
        "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
    TORCH_CHECK(
        !query.is_nested() && !key.is_nested(),
        "_scaled_dot_product_attention: Nested tensors for query / key are not supported when is_causal=True");

    // Replace attn_mask with causal mask; lower triangular elements take part
    // in attention.
    const auto L = query.sym_size(-2), S = key.sym_size(-2);
    attn_mask =
        at::ones_symint({L, S}, query.options().dtype(at::kBool)).tril();
    attn_mask = sdp::convert_boolean_attn_mask(attn_mask, query.dtype());
  }
  auto attn = at::matmul(query, key.transpose(-2, -1) * scaling_factor);
  if (attn_mask.has_value()) {
    if (at::areAnyTensorSubclassLike({attn, *attn_mask})) {
      attn = attn.add(*attn_mask);
    } else {
      attn.add_(*attn_mask);
    }
  }
  attn = at::softmax(attn, -1);
  if (dropout_p > 0.0) {
    if (dropout_mask.has_value()) {
      // In order to validate the correctness of the fused kernels, we need to
      // use the same dropout mask in order to compare the results.
      TORCH_WARN_ONCE("Dropout mask should only be used for testing purposes.");
      attn = attn.masked_fill(dropout_mask->logical_not(), 0.0);
      auto dropout_scaling = 1.0 / (1 - dropout_p);
      return std::make_tuple(at::matmul(attn, value * dropout_scaling), attn);
    } else {
      attn = at::dropout(attn, dropout_p, true);
    }
  }

  return std::make_tuple(at::matmul(attn, value), attn);
}

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    const c10::optional<Tensor>& dropout_mask,
    c10::optional<double> scale) {
  // on ATSM, the efficient_attention path is not available
  // With naive math path, oneDNN matmul has overflow issue with fp16 inputs
  // as a WA, convert fp16 inputs to fp32
  return IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      query_.scalar_type(),
      "scaled_dot_product_attention_math",
      [&] {
        bool is_half = std::is_same<scalar_t, at::Half>::value;
        if (is_half) {
          Tensor query_fp32 = query_.to(at::kFloat);
          Tensor key_fp32 = key.to(at::kFloat);
          Tensor value_fp32 = value.to(at::kFloat);
          return _scaled_dot_product_attention_math_impl(
              query_fp32,
              key_fp32,
              value_fp32,
              attn_mask_,
              dropout_p,
              is_causal,
              dropout_mask,
              scale);
        }
        return _scaled_dot_product_attention_math_impl(
            query_,
            key,
            value,
            attn_mask_,
            dropout_p,
            is_causal,
            dropout_mask,
            scale);
      });
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
_scaled_dot_product_efficient_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<at::Tensor>& attn_bias,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale) {
  auto out = _scaled_dot_product_efficient_attention_impl(
      query, key, value, attn_bias, is_causal, dropout_p, scale);
  auto softmax_lse = at::empty(
      {query.size(0), query.size(1), query.size(2)},
      query.options().dtype(at::kFloat));
  Tensor seed_t = at::empty({}, at::dtype(at::kLong).device(at::kXPU));
  Tensor offset_t = at::empty({}, at::dtype(at::kLong).device(at::kXPU));
  return std::make_tuple(
      std::move(out),
      std::move(softmax_lse),
      std::move(seed_t),
      std::move(offset_t));
}

inline bool xetla_supported(
    Tensor q,
    Tensor k,
    Tensor v,
    bool is_training,
    const c10::optional<Tensor>& b) {
  bool is_supported = false;
#if defined(USE_XETLA)
  if (dpcppGetDeviceHasXMX()) {
    DeviceId curDevID;
    AT_DPCPP_CHECK(dpcppGetDevice(&curDevID));
    bool bias_support = true;
    // bias size requires [batch_size, 1, q_len, k_len]
    if (b.has_value()) {
      if (b.value().size(0) != q.size(0))
        bias_support = false;
      if (b.value().size(1) != 1)
        bias_support = false;
      if (b.value().size(2) != q.size(2))
        bias_support = false;
    }
    if (q.dtype() == at::kHalf && k.dtype() == at::kHalf &&
        v.dtype() == at::kHalf && !is_training &&
        Settings::I().has_2d_block_array(curDevID) && bias_support == true) {
      if ((q.size(-1) * sizeof(at::Half) % 128 == 0) &&
          (v.size(-1) * sizeof(at::Half) % 128 == 0))
        is_supported = true;
    }
  }
#endif
  return is_supported;
}

int64_t _fused_sdp_choice(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale) {
  bool is_training =
      (query.requires_grad() || key.requires_grad() || value.requires_grad());
  // We have implemented efficient_attention backend with xetla,
  // flash_attention backend is not supported now, which will be implemented
  // in the future. So we provide two backends here.
  sdp::SDPBackend backend =
      xetla_supported(query, key, value, is_training, attn_mask_)
      ? sdp::SDPBackend::efficient_attention
      : sdp::SDPBackend::math;
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
    bool is_causal,
    c10::optional<double> scale) {
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

  TORCH_CHECK(
      query.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");
  TORCH_CHECK(
      key.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");
  TORCH_CHECK(
      value.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");

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

#if defined(USE_XETLA)
  TORCH_CHECK(
      dpcppGetDeviceHasXMX(),
      "SDP kernel requires XMX, but the current platform has no XMX ...");
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
#else
  AT_ERROR("SDP: xetla library not found in compilation");
#endif
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
// *output      shape  : [bs * beam, num_head, kv_in_len + kv_out_len,
// head_dim]
//              layout : [bs * beam, kv_in_len + kv_out_len, num_head,
//              head_dim]
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

  TORCH_CHECK(
      query.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");
  TORCH_CHECK(
      key.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");
  TORCH_CHECK(
      value.scalar_type() == at::kHalf, "IPEX SDP only supports half datatype");

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
  TORCH_CHECK(
      beam_width == 1 || beam_width == 4,
      "SDP only support greedy search and beam search with beam size is 1 or 4");
  uint32_t num_keys_in = key.size(2);
  uint32_t num_keys_out = key_cache.size(2);
  auto output = at::empty_like(query);
  auto dpcpp_queue = dpcppGetCurrentQueue();
  RECORD_FUNCTION("xetla_fsdp_index_forward", {});

#if defined(USE_XETLA)
  TORCH_CHECK(
      dpcppGetDeviceHasXMX(),
      "SDP kernel requires XMX, but the current platform has no XMX ...");
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
#else
  AT_ERROR("SDP: xetla library not found in compilation");
#endif
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
