#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/record_function.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <CL/sycl.hpp>
#include <runtime/Device.h>
#include <runtime/Utils.h>
#include <torch/autograd.h>
#include <torch/custom_class.h>
#include <utils/DPCPP.h>
#include <utils/SimpleTrace.h>
#include "../Blas.h"
#include "../DistributionTemplates.h"
#include "../RandomEngine.h"
#include "../comm/ATDispatch.h"
#include "../comm/AccumulateType.h"
#include "dropout.h"
#include "sdp_utils.h"
#include "utils/CustomOperatorRegistration.h"

using namespace torch::autograd;
namespace at {
namespace AtenIpexTypeXPU {

std::tuple<Tensor, Tensor, Tensor, Tensor> ipex_sdp_dropout_backward(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_bias,
    const Tensor& out,
    const Tensor& logsumexp,
    const Tensor& dropout_mask,
    double dropout_p,
    bool grad_input_mask,
    bool causal,
    c10::optional<double> scale);

inline Tensor _scaled_dot_product_efficient_attention_impl(
    const Tensor& _query,
    const Tensor& _key,
    const Tensor& _value,
    const c10::optional<Tensor>& attn_mask,
    const c10::optional<at::Tensor>& dropout_mask,
    const c10::optional<at::Tensor>& seed_t,
    const c10::optional<at::Tensor>& offset_t,
    Tensor& softmax_lse,
    bool is_causal,
    bool is_training,
    double dropout_p,
    c10::optional<double> scale) {
#if defined(USE_XETLA)
  // check attn_mask padded
  uint32_t attn_mask_padded_block_size = 0;
  if (attn_mask.has_value()) {
    std::vector<int64_t> sz = attn_mask->sizes().vec();
    int64_t lastDim = sz[sz.size() - 1];
    int64_t alignTo = 16;
    attn_mask_padded_block_size = alignTo * ((lastDim + alignTo - 1) / alignTo);
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

  const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;
  auto xeType = sdp::aten_to_Xetla_dtype(query);
  gpu_arch xeArch = sdp::aten_to_Xetla_arch();
  auto cgfs = gpu::xetla::fmha_forward_kernel(
      xeArch,
      xeType,
      {query.data_ptr(),
       key.data_ptr(),
       value.data_ptr(),
       /* alibi */ nullptr,
       attn_mask.has_value() ? attn_mask->data_ptr() : (void*)nullptr,
       dropout_mask.has_value() ? dropout_mask->data_ptr() : (void*)nullptr,
       output.data_ptr(),
       softmax_lse.data_ptr(),
       softmax_scale,
       /* beta */ 1.0f,
       dropout_p,
       query.size(0),
       query.size(1),
       key.size(1),
       query.size(3),
       query.size(2),
       key.size(2),
       attn_mask.has_value() ? attn_mask->stride(0) : -1,
       attn_mask.has_value() ? attn_mask->stride(1) : -1,
       attn_mask.has_value() ? attn_mask->stride(2) : -1,
       /* ablibi padded size */ 0,
       attn_mask_padded_block_size,
       is_causal,
       false,
       is_training,
       use_dropout,
       seed_t.has_value() ? (uint64_t)*seed_t.value().data_ptr<int64_t>() : -1,
       offset_t.has_value() ? (uint64_t)*offset_t.value().data_ptr<int64_t>()
                            : -1});
  DPCPP_Q_SUBMIT_CGFS(dpcpp_queue, cgfs);

  return output;
#else
  AT_ERROR("SDP: xetla library not found in compilation");
  // TODO: sycl kernel impl for efficient_attention
  // auto result = naive_scaled_dot_product(query, key, value, is_causal);
  // return std::forward_as_tuple(std::get<0>(result), std::get<1>(result));
#endif
}

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math_native_impl(
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
      sdp::native_calculate_scale(
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

  // [Original] Scale q, k before matmul for stability see
  // https://tinyurl.com/sudb9s96 for math
  // Here we apply scaling after matmul for op fusion purpose
  bool is_negative_scaling = scale.has_value() && scale.value() < 0.0;
  const auto orig_scaling_factor = sdp::calculate_scale(
      query_, is_negative_scaling ? std::abs(scale.value()) : scale);

  if (is_causal) {
    TORCH_CHECK(
        !attn_mask.has_value(),
        "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
    TORCH_CHECK(
        !query_.is_nested() && !key.is_nested(),
        "_scaled_dot_product_attention: Nested tensors for query / key are not supported when is_causal=True");

    // Replace attn_mask with causal mask; lower triangular elements take part
    // in attention.
    const auto L = query_.sym_size(-2), S = key.sym_size(-2);
    attn_mask =
        at::ones_symint({L, S}, query_.options().dtype(at::kBool)).tril();
    attn_mask = sdp::convert_boolean_attn_mask(attn_mask, query_.dtype());
  }

  Tensor attn;
  if (attn_mask.has_value()) {
    attn_mask = attn_mask->contiguous();
    if (is_negative_scaling) {
      attn = trans_matmul_div_add(
          key,
          /*dim1=*/-1,
          /*dim2=*/-1,
          query_,
          c10::SymFloat(0.0) - orig_scaling_factor,
          *attn_mask,
          1.0);
    } else {
      attn = trans_matmul_div_add(
          key,
          /*dim1=*/-1,
          /*dim2=*/-1,
          query_,
          orig_scaling_factor,
          *attn_mask,
          1.0);
    }
  } else {
    if (is_negative_scaling) {
      attn = trans_matmul_div_scalar(
          key,
          /*dim1=*/-1,
          /*dim2=*/-1,
          query_,
          c10::SymFloat(0.0) - orig_scaling_factor);
    } else {
      attn = trans_matmul_div_scalar(
          key, /*dim1=*/-1, /*dim2=*/-1, query_, orig_scaling_factor);
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
  if (query_.requires_grad() || key.requires_grad() || value.requires_grad()) {
    return IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        query_.scalar_type(),
        "scaled_dot_product_attention_math",
        [&] {
          bool is_half = std::is_same<scalar_t, at::Half>::value;
          if (is_half) {
            c10::optional<Tensor> attn_mask_fp32;
            Tensor query_fp32 = query_.to(at::kFloat);
            Tensor key_fp32 = key.to(at::kFloat);
            Tensor value_fp32 = value.to(at::kFloat);
            if (attn_mask_.has_value()) {
              attn_mask_fp32 = attn_mask_.value().to(at::kFloat);
            } else {
              attn_mask_fp32 = attn_mask_;
            }
            auto [attn_output, attn_weight] =
                _scaled_dot_product_attention_math_native_impl(
                    query_fp32,
                    key_fp32,
                    value_fp32,
                    attn_mask_fp32,
                    dropout_p,
                    is_causal,
                    dropout_mask,
                    scale);
            return std::make_tuple(
                attn_output.to(at::kHalf), attn_weight.to(at::kHalf));
          }
          return _scaled_dot_product_attention_math_native_impl(
              query_,
              key,
              value,
              attn_mask_,
              dropout_p,
              is_causal,
              dropout_mask,
              scale);
        });
  } else {
    // accelerate for inference
    return IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        query_.scalar_type(),
        "scaled_dot_product_attention_math",
        [&] {
          bool is_half = std::is_same<scalar_t, at::Half>::value;
          if (is_half) {
            c10::optional<Tensor> attn_mask_fp32;
            Tensor query_fp32 = query_.to(at::kFloat);
            Tensor key_fp32 = key.to(at::kFloat);
            Tensor value_fp32 = value.to(at::kFloat);
            if (attn_mask_.has_value()) {
              attn_mask_fp32 = attn_mask_.value().to(at::kFloat);
            } else {
              attn_mask_fp32 = attn_mask_;
            }
            auto [attn_output, attn_weight] =
                _scaled_dot_product_attention_math_impl(
                    query_fp32,
                    key_fp32,
                    value_fp32,
                    attn_mask_fp32,
                    dropout_p,
                    is_causal,
                    dropout_mask,
                    scale);
            return std::make_tuple(
                attn_output.to(at::kHalf), attn_weight.to(at::kHalf));
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
  int64_t B = query.size(0);
  int64_t num_heads = query.size(1);
  int64_t M = query.size(-2);
  int64_t N = key.size(-2);

  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      c10::nullopt, at::xpu::detail::getDefaultXPUGenerator());
  std::pair<uint64_t, uint64_t> philox_state;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    philox_state = gen->philox_engine_inputs(B * num_heads * M * N);
  }
  PhiloxState rng_engine_inputs(
      std::get<0>(philox_state), std::get<1>(philox_state));
  auto [seed, offset] = philox_unpack(rng_engine_inputs);
  Tensor seed_t = at::scalar_tensor(
      at::Scalar(static_cast<int64_t>(seed)), at::dtype(at::kLong));
  Tensor offset_t = at::scalar_tensor(
      at::Scalar(static_cast<int64_t>(offset)), at::dtype(at::kLong));

  auto softmax_lse = at::empty(
      {query.size(0), query.size(1), query.size(2)},
      query.options().dtype(at::kFloat));

  auto out = _scaled_dot_product_efficient_attention_impl(
      query,
      key,
      value,
      attn_bias,
      c10::nullopt,
      seed_t,
      offset_t,
      softmax_lse,
      is_causal,
      compute_log_sumexp,
      dropout_p,
      scale);
  return std::make_tuple(
      std::move(out),
      std::move(softmax_lse),
      std::move(seed_t),
      std::move(offset_t));
}

std::tuple<Tensor, Tensor, Tensor> ipex_sdp_dropout_forward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<at::Tensor>& attn_bias,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale) {
  RECORD_FUNCTION("ipex_sdp_dropout_forward", {});
  int64_t B = query.size(0);
  int64_t num_heads = query.size(1);
  int64_t M = query.size(-2);
  int64_t N = key.size(-2);
  const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;
  Tensor dropout_mask = at::empty(
      {B, num_heads, M, N},
      query.options().dtype(c10::CppTypeToScalarType<uint8_t>::value));
  if (use_dropout) {
    dropout_mask = at::AtenIpexTypeXPU::dropout_mask_only<uint8_t>(
        dropout_mask, dropout_p);
  }
  auto softmax_lse = at::empty(
      {query.size(0), query.size(1), query.size(2)},
      query.options().dtype(at::kFloat));

  auto out = _scaled_dot_product_efficient_attention_impl(
      query,
      key,
      value,
      attn_bias,
      dropout_mask,
      c10::nullopt,
      c10::nullopt,
      softmax_lse,
      is_causal,
      compute_log_sumexp,
      dropout_p,
      scale);
  return std::make_tuple(
      std::move(out), std::move(softmax_lse), std::move(dropout_mask));
}

class IPEXSDPDropoutOp : public Function<IPEXSDPDropoutOp> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      const Tensor& query,
      const Tensor& key,
      const Tensor& value,
      const c10::optional<at::Tensor>& attn_bias,
      bool compute_log_sumexp,
      double dropout_p,
      bool is_causal,
      c10::optional<double> scale) {
#ifdef BUILD_SIMPLE_TRACE
    SimpleTrace trace(
        "IPEXSDPDropoutOp forward -> at::AtenIpexTypeXPU::IPEXSDPDropoutOp::forward");
#endif
    ctx->saved_data["dropout_p"] = dropout_p;
    ctx->saved_data["is_causal"] = is_causal;
    ctx->saved_data["scale"] = scale;
    ctx->saved_data["attn_bias"] = attn_bias;
    ctx->saved_data["attn_bias_requires_grad"] =
        attn_bias.has_value() ? attn_bias.value().requires_grad() : false;

    auto outputs = ipex_sdp_dropout_forward(
        query,
        key,
        value,
        attn_bias,
        compute_log_sumexp,
        dropout_p,
        is_causal,
        scale);
    ctx->save_for_backward(
        {query,
         key,
         value,
         std::get<0>(outputs),
         std::get<1>(outputs),
         std::get<2>(outputs)});
    variable_list result = {
        std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs)};
    return result;
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_outputs) {
#ifdef BUILD_SIMPLE_TRACE
    SimpleTrace trace(
        "IPEXSDPDropoutOp backward -> at::AtenIpexTypeXPU::IPEXSDPDropoutOp::backward");
#endif
    auto attn_bias = ctx->saved_data["attn_bias"].toOptional<at::Tensor>();
    auto dropout_p = ctx->saved_data["dropout_p"].toDouble();
    auto is_causal = ctx->saved_data["is_causal"].toBool();
    auto scale = ctx->saved_data["scale"].toOptional<double>();
    auto compute_grad = ctx->saved_data["attn_bias_requires_grad"].toBool();
    auto saved = ctx->get_saved_variables();
    Tensor query = saved[0];
    Tensor key = saved[1];
    Tensor value = saved[2];
    Tensor output = saved[3];
    Tensor logsumexp = saved[4];
    Tensor dropout_mask = saved[5];

    auto grad_inputs = ipex_sdp_dropout_backward(
        grad_outputs[0],
        query,
        key,
        value,
        attn_bias,
        output,
        logsumexp,
        dropout_mask,
        dropout_p,
        compute_grad,
        is_causal,
        scale);
    return {
        std::get<0>(grad_inputs),
        std::get<1>(grad_inputs),
        std::get<2>(grad_inputs),
        std::get<3>(grad_inputs),
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor()};
  }
};

template <int alignment>
bool is_aligned(const SymInt& size) {
  return size % alignment == 0;
}

template <int alignment>
at::Tensor pad_bias(const at::Tensor& attn_bias) {
  auto last_dim_size = attn_bias.sym_size(-1);
  auto pad_count = alignment - (last_dim_size % alignment);
  auto padded_bias = at::pad_symint(attn_bias, {c10::SymInt(0), pad_count});
  return padded_bias.slice_symint(-1, 0, last_dim_size);
}

Tensor preprocess_mask(
    const Tensor& mask,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  constexpr int mem_eff_alignment = 16;
  // Expand to 4d case
  at::Tensor attn_mask = mask.expand_symint(
      {query.sym_size(0),
       query.sym_size(1),
       query.sym_size(2),
       key.sym_size(2)});

  bool aligned_last_dim = is_aligned<mem_eff_alignment>(attn_mask.sym_size(-1));
  // Apply pad_bias and store the result in attn_mask
  if (!aligned_last_dim) {
    return pad_bias<mem_eff_alignment>(attn_mask);
  }
  // Check and make the tensor contiguous if needed
  if (attn_mask.sym_stride(0) % 16 != 0 || attn_mask.sym_stride(1) % 16 != 0 ||
      attn_mask.sym_stride(2) % 16 != 0) {
    return attn_mask.contiguous();
  }

  return attn_mask;
}

// We compute dropout mask tensor then pass to forward, and save for backward
Tensor xetla_sdp_dropout(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<at::Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale) {
  c10::optional<Tensor> attn_mask =
      sdp::convert_boolean_attn_mask(attn_mask_, query_.dtype());
  bool compute_logsumexp =
      (query_.requires_grad() || key.requires_grad() || value.requires_grad());
  if (attn_mask.has_value()) {
    attn_mask.value() = preprocess_mask(attn_mask.value(), query_, key, value);
    ;
  }
  auto out_and_lse = IPEXSDPDropoutOp::apply(
      query_,
      key,
      value,
      attn_mask,
      compute_logsumexp,
      dropout_p,
      is_causal,
      scale);
  return out_and_lse[0];
}

int64_t _fused_sdp_choice(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale) {
  // We have implemented efficient_attention backend with xetla, flash_attention
  // backend is not supported now, which will be implemented in the future. So
  // we provide two backends here.
  sdp::sdp_params kernel_params{
      query, key, value, attn_mask_, dropout_p, is_causal};
  sdp::SDPBackend backend = sdp::use_mem_efficient_attention(kernel_params)
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

  int64_t B = query.size(0);
  int64_t num_heads_q = query.size(1);
  int64_t num_heads_k = key.size(1);
  int64_t head_dim = query.size(3);
  int64_t M = query.size(-2);
  int64_t N = key.size(-2);

  auto output = at::empty_like(query);
  auto dpcpp_queue = dpcppGetCurrentQueue();
  char str__[100];
  sprintf(
      str__,
      "xetla_fsdp_forward_atten_mask_alibi_strided(Nq=%ld, Nkv=%ld, M=%ld, N=%ld)",
      num_heads_q,
      num_heads_k,
      M,
      N);
  RECORD_FUNCTION(str__, {});

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
  Tensor attn_mask_bc;
  if (attn_mask.has_value()) {
    attn_mask_padded_block_size = attn_mask.value().size(-1);
    // align PyTorch mask preprocess (broadcast without memory change)
    // TODO: align padding strategy
    attn_mask_bc = attn_mask.value().expand(
        {query.size(0),
         query.size(1),
         query.size(2),
         attn_mask_padded_block_size});
    TORCH_CHECK(
        (attn_mask_padded_block_size * key.itemsize() % 8 == 0),
        "XeTLA SDP Attention mask needs 8bytes aligned on leading dimension ...");
  }
  auto softmax_lse = at::empty({}, query.options().dtype(at::kFloat));

#if defined(USE_XETLA)
  auto xeType = sdp::aten_to_Xetla_dtype(query);
  gpu_arch xeArch = sdp::aten_to_Xetla_arch();
  auto cgfs = gpu::xetla::fmha_forward_kernel(
      xeArch,
      xeType,
      {query.data_ptr(),
       key.data_ptr(),
       value.data_ptr(),
       alibi.has_value() ? alibi.value().data_ptr() : (void*)nullptr,
       attn_mask.has_value() ? attn_mask_bc.data_ptr() : (void*)nullptr,
       nullptr,
       output.data_ptr(),
       softmax_lse.data_ptr(),
       alpha,
       beta,
       dropout_p,
       B,
       num_heads_q,
       num_heads_k,
       head_dim,
       M,
       N,
       attn_mask.has_value() ? attn_mask_bc.stride(0) : -1,
       attn_mask.has_value() ? attn_mask_bc.stride(1) : -1,
       attn_mask.has_value() ? attn_mask_bc.stride(2) : -1,
       alibi_padded_block_size,
       attn_mask_padded_block_size,
       is_causal,
       seq_last,
       false, // is_training
       false, // use_dropout
       (int)0,
       (int)0});
  DPCPP_Q_SUBMIT_CGFS(dpcpp_queue, cgfs);
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
  bool is_broadcast = false;
  if (attn_mask.has_value()) {
    attn_mask_padding = attn_mask.value().size(-1);
    TORCH_CHECK(
        attn_mask->size(0) == query.size(0) &&
            attn_mask->size(2) == query.size(2),
        "unsupported attention mask size");
    TORCH_CHECK(
        attn_mask->size(1) == 1 || attn_mask->size(1) == query.size(1),
        "SDP index only supports attn_mask second dim with size 1 or num heads");
    TORCH_CHECK(
        (attn_mask_padding * key.itemsize() % 8 == 0),
        "XeTLA SDP Attention mask needs 8bytes aligned on leading dimension ...");
    if (attn_mask->size(1) == query.size(1))
      is_broadcast = true;
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

#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  TORCH_CHECK(
      dpcppGetDeviceHasXMX(),
      "SDP kernel requires XMX, but the current platform has no XMX ...");
  auto cgfs = gpu::xetla::fmha_forward_index_kernel(
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
      is_causal,
      is_broadcast);
  DPCPP_Q_SUBMIT_CGFS(dpcpp_queue, cgfs);
#else
  AT_ERROR("SDP: xetla library not found in compilation");
#endif
  return output;
}

template <typename scalar_t>
void xetla_paged_attention_impl_v1(
    Tensor& out,
    const Tensor& query,
    const Tensor& key_cache,
    const Tensor& value_cache,
    const Tensor& head_mapping,
    const Tensor& block_tables,
    const Tensor& context_lens,
    const double head_scale,
    const int64_t block_size,
    const int64_t max_context_len,
    const c10::optional<Tensor>& alibi_slopes) {
  uint32_t num_seqs = query.size(0);
  uint32_t num_heads = query.size(1);
  uint32_t head_size = query.size(2);
  uint32_t num_kv_heads = key_cache.size(1);
  uint32_t max_num_blocks_per_seq = block_tables.size(1);

  // TODO(zw): alibi_slopes is optional, not used currently.
  const float* alibi_slopes_ptr = alibi_slopes
      ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
      : nullptr;

  auto dpcpp_queue = dpcppGetCurrentQueue();
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  auto cgfs = gpu::xetla::paged_attention_v1(
      reinterpret_cast<scalar_t*>(out.data_ptr()),
      reinterpret_cast<scalar_t*>(query.data_ptr()),
      reinterpret_cast<scalar_t*>(key_cache.data_ptr()),
      reinterpret_cast<scalar_t*>(value_cache.data_ptr()),
      head_mapping.data_ptr<int32_t>(),
      block_tables.data_ptr<int32_t>(),
      context_lens.data_ptr<int32_t>(),
      head_scale,
      num_seqs,
      num_heads,
      num_kv_heads,
      head_size,
      block_size,
      max_num_blocks_per_seq,
      max_context_len);
  DPCPP_Q_SUBMIT_CGFS(dpcpp_queue, cgfs);

#else
  AT_ERROR("PagedAttention: xetla library not found in compilation");
#endif
}

void xetla_paged_attention_v1(
    Tensor& out,
    const Tensor& query,
    const Tensor& key_cache,
    const Tensor& value_cache,
    const Tensor& head_mapping,
    const Tensor& block_tables,
    const Tensor& context_lens,
    const double head_scale,
    const int64_t block_size,
    const int64_t max_context_len,
    const c10::optional<Tensor>& alibi_slopes) {
  RECORD_FUNCTION("xetla_paged_attention_v1", {});

  if (out.scalar_type() == at::kHalf) {
    xetla_paged_attention_impl_v1<sycl::half>(
        out,
        query,
        key_cache,
        value_cache,
        head_mapping,
        block_tables,
        context_lens,
        head_scale,
        block_size,
        max_context_len,
        alibi_slopes);
  } else {
    AT_ERROR("PagedAttention: only support half");
  }
}

template <typename scalar_t>
void xetla_paged_attention_impl_v2(
    Tensor& max_logits,
    Tensor& exp_sums,
    Tensor& tmp_out,
    Tensor& out,
    const Tensor& query,
    const Tensor& key_cache,
    const Tensor& value_cache,
    const Tensor& head_mapping,
    const Tensor& block_tables,
    const Tensor& context_lens,
    const double head_scale,
    const int64_t block_size,
    const int64_t max_context_len,
    const c10::optional<Tensor>& alibi_slopes) {
  uint32_t num_seqs = query.size(0);
  uint32_t num_heads = query.size(1);
  uint32_t head_size = query.size(2);
  uint32_t num_kv_heads = key_cache.size(1);
  uint32_t max_num_blocks_per_seq = block_tables.size(1);

  // TODO(zw): alibi_slopes is optional, not used currently.
  const float* alibi_slopes_ptr = alibi_slopes
      ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
      : nullptr;

  auto dpcpp_queue = dpcppGetCurrentQueue();
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  auto cgfs = gpu::xetla::paged_attention_v2(
      max_logits.data_ptr<float>(),
      exp_sums.data_ptr<float>(),
      reinterpret_cast<scalar_t*>(tmp_out.data_ptr()),
      reinterpret_cast<scalar_t*>(out.data_ptr()),
      reinterpret_cast<scalar_t*>(query.data_ptr()),
      reinterpret_cast<scalar_t*>(key_cache.data_ptr()),
      reinterpret_cast<scalar_t*>(value_cache.data_ptr()),
      head_mapping.data_ptr<int32_t>(),
      block_tables.data_ptr<int32_t>(),
      context_lens.data_ptr<int32_t>(),
      head_scale,
      num_seqs,
      num_heads,
      num_kv_heads,
      head_size,
      block_size,
      max_num_blocks_per_seq,
      max_context_len);
  DPCPP_Q_SUBMIT_CGFS(dpcpp_queue, cgfs);
#else
  AT_ERROR("PagedAttention: xetla library not found in compilation");
#endif
}

void xetla_paged_attention_v2(
    Tensor& max_logits,
    Tensor& exp_sums,
    Tensor& tmp_out,
    Tensor& out,
    const Tensor& query,
    const Tensor& key_cache,
    const Tensor& value_cache,
    const Tensor& head_mapping,
    const Tensor& block_tables,
    const Tensor& context_lens,
    const double head_scale,
    const int64_t block_size,
    const int64_t max_context_len,
    const c10::optional<Tensor>& alibi_slopes) {
  RECORD_FUNCTION("xetla_paged_attention_v2", {});

  if (out.scalar_type() == at::kHalf) {
    xetla_paged_attention_impl_v2<sycl::half>(
        max_logits,
        exp_sums,
        tmp_out,
        out,
        query,
        key_cache,
        value_cache,
        head_mapping,
        block_tables,
        context_lens,
        head_scale,
        block_size,
        max_context_len,
        alibi_slopes);
  } else {
    AT_ERROR("PagedAttention: only support half");
  }
}

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeNestedTensorXPU {
static constexpr int TRANSFORM_BIAS_RESCALE_VEC = 4;

namespace impl {

template <typename scalar_t, typename accscalar_t>
struct transform_bias_rescale_qkv_add_padding_impl {
  void operator()(sycl::nd_item<1> item) const {
    const auto NH = q_k_v.size(2);
    const auto T = q_k_v.size(3);
    const auto DH = q_k_v.size(4);

    const auto stride_0 = q_k_v.stride(0);
    const auto stride_1 = q_k_v.stride(1);
    const auto stride_2 = q_k_v.stride(2);
    const auto stride_3 = q_k_v.stride(3);
    const auto stride_4 = q_k_v.stride(4);
    scalar_t* const data = q_k_v.data();

    const int32_t local_id = item.get_local_id(0);
    const int32_t global_id = item.get_group(0);

    const auto t = global_id % T;
    const auto b = global_id / T;

    const auto D = NH * DH;
    const auto _3D = 3 * D;

    const auto offset_for_batch = offsets[b];
    const auto input_dim = 1;
    const auto* sizes_i = input_sizes + b * input_dim;

    if (assume_aligned) {
      constexpr int VEC = TRANSFORM_BIAS_RESCALE_VEC;
      using LoadT = at::native::Memory::aligned_vector_loop<scalar_t, VEC>;
      for (int32_t d_v = local_id; d_v < D / VEC;
           d_v += item.get_local_range(0)) {
        auto d = d_v * VEC;
        auto nh = d / DH;
        auto dh = d % DH;
        scalar_t qkv_bias_q[VEC];
        scalar_t qkv_bias_k[VEC];
        scalar_t qkv_bias_v[VEC];
        scalar_t qkv_q[VEC];
        scalar_t qkv_k[VEC];
        scalar_t qkv_v[VEC];

        const auto first_item_offset = t * _3D + d;
        const auto last_item_offset = first_item_offset + VEC - 1;
        const bool first_item_in_bounds = first_item_offset < sizes_i[0];
        const bool entire_vec_in_bounds = last_item_offset < sizes_i[0];

        // Here we require D % VEC == 0 for these vectorized loads.
        *reinterpret_cast<LoadT*>(&qkv_bias_q) =
            *reinterpret_cast<const LoadT*>(&qkv_bias[d + 0 * D]);
        *reinterpret_cast<LoadT*>(&qkv_bias_k) =
            *reinterpret_cast<const LoadT*>(&qkv_bias[d + 1 * D]);
        *reinterpret_cast<LoadT*>(&qkv_bias_v) =
            *reinterpret_cast<const LoadT*>(&qkv_bias[d + 2 * D]);

        if (entire_vec_in_bounds) {
          const auto offset = offset_for_batch + first_item_offset;
          *reinterpret_cast<LoadT*>(&qkv_q) =
              *reinterpret_cast<const LoadT*>(&qkv[offset + 0 * D]);
          *reinterpret_cast<LoadT*>(&qkv_k) =
              *reinterpret_cast<const LoadT*>(&qkv[offset + 1 * D]);
          *reinterpret_cast<LoadT*>(&qkv_v) =
              *reinterpret_cast<const LoadT*>(&qkv[offset + 2 * D]);

#pragma unroll
          for (auto ii = 0; ii < VEC; ++ii) {
            qkv_q[ii] = static_cast<scalar_t>(
                (static_cast<accscalar_t>(qkv_q[ii]) +
                 static_cast<accscalar_t>(qkv_bias_q[ii])) *
                static_cast<accscalar_t>(inv_sqrt_dim_per_head));
            qkv_k[ii] = static_cast<scalar_t>(
                (static_cast<accscalar_t>(qkv_k[ii]) +
                 static_cast<accscalar_t>(qkv_bias_k[ii])));
            qkv_v[ii] = static_cast<scalar_t>(
                (static_cast<accscalar_t>(qkv_v[ii]) +
                 static_cast<accscalar_t>(qkv_bias_v[ii])));
          }
        } else if (first_item_in_bounds) {
          const auto offset = offset_for_batch + first_item_offset;
          qkv_q[0] = qkv[offset + 0 * D];
          qkv_k[0] = qkv[offset + 1 * D];
          qkv_v[0] = qkv[offset + 2 * D];
          qkv_q[0] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_q[0]) +
               static_cast<accscalar_t>(qkv_bias_q[0])) *
              static_cast<accscalar_t>(inv_sqrt_dim_per_head));
          qkv_k[0] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_k[0]) +
               static_cast<accscalar_t>(qkv_bias_k[0])));
          qkv_v[0] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_v[0]) +
               static_cast<accscalar_t>(qkv_bias_v[0])));
#pragma unroll
          for (auto ii = 1; ii < VEC; ++ii) {
            const auto loop_offset = offset + ii;
            if (loop_offset < sizes_i[0]) {
              qkv_q[ii] = qkv[loop_offset + 0 * D];
              qkv_k[ii] = qkv[loop_offset + 1 * D];
              qkv_v[ii] = qkv[loop_offset + 2 * D];
              qkv_q[ii] = static_cast<scalar_t>(
                  (static_cast<accscalar_t>(qkv_q[ii]) +
                   static_cast<accscalar_t>(qkv_bias_q[ii])) *
                  static_cast<accscalar_t>(inv_sqrt_dim_per_head));
              qkv_k[ii] = static_cast<scalar_t>(
                  (static_cast<accscalar_t>(qkv_k[ii]) +
                   static_cast<accscalar_t>(qkv_bias_k[ii])));
              qkv_v[ii] = static_cast<scalar_t>(
                  (static_cast<accscalar_t>(qkv_v[ii]) +
                   static_cast<accscalar_t>(qkv_bias_v[ii])));
            } else {
              qkv_q[ii] = 0;
              qkv_k[ii] = 0;
              qkv_v[ii] = 0;
            }
          }
        } else {
#pragma unroll
          for (auto ii = 0; ii < VEC; ++ii) {
            qkv_q[ii] = 0;
            qkv_k[ii] = 0;
            qkv_v[ii] = 0;
          }
        }

        // Here we require DH % VEC == 0 for these vectorized stores.
        *reinterpret_cast<LoadT*>(
            &data
                [0 * stride_0 + b * stride_1 + nh * stride_2 + t * stride_3 +
                 dh * stride_4]) = *reinterpret_cast<const LoadT*>(&qkv_q);
        *reinterpret_cast<LoadT*>(
            &data
                [1 * stride_0 + b * stride_1 + nh * stride_2 + t * stride_3 +
                 dh * stride_4]) = *reinterpret_cast<const LoadT*>(&qkv_k);
        *reinterpret_cast<LoadT*>(
            &data
                [2 * stride_0 + b * stride_1 + nh * stride_2 + t * stride_3 +
                 dh * stride_4]) = *reinterpret_cast<const LoadT*>(&qkv_v);
      }
    } else {
      for (int32_t d = local_id; d < D; d += item.get_local_range(0)) {
        auto nh = d / DH;
        auto dh = d % DH;
        scalar_t qkv_bias_q = qkv_bias[d + 0 * D];
        scalar_t qkv_bias_k = qkv_bias[d + 1 * D];
        scalar_t qkv_bias_v = qkv_bias[d + 2 * D];

        const auto item_offset = t * _3D + d;
        const bool in_bounds = item_offset < sizes_i[0];
        scalar_t qkv_q, qkv_k, qkv_v;

        if (in_bounds) {
          const auto qkv_offset = offset_for_batch + item_offset;
          qkv_q = qkv[qkv_offset + 0 * D];
          qkv_k = qkv[qkv_offset + 1 * D];
          qkv_v = qkv[qkv_offset + 2 * D];
          qkv_q = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_q) +
               static_cast<accscalar_t>(qkv_bias_q)) *
              static_cast<accscalar_t>(inv_sqrt_dim_per_head));
          qkv_k = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_k) +
               static_cast<accscalar_t>(qkv_bias_k)));
          qkv_v = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_v) +
               static_cast<accscalar_t>(qkv_bias_v)));
        } else {
          qkv_q = 0;
          qkv_k = 0;
          qkv_v = 0;
        }

        data
            [0 * stride_0 + b * stride_1 + nh * stride_2 + t * stride_3 +
             dh * stride_4] = qkv_q;
        data
            [1 * stride_0 + b * stride_1 + nh * stride_2 + t * stride_3 +
             dh * stride_4] = qkv_k;
        data
            [2 * stride_0 + b * stride_1 + nh * stride_2 + t * stride_3 +
             dh * stride_4] = qkv_v;
      }
    }
  }

  transform_bias_rescale_qkv_add_padding_impl(
      const PackedTensorAccessor64<scalar_t, 1> qkv_,
      const PackedTensorAccessor64<scalar_t, 1> qkv_bias_,
      const int* offsets_,
      const int* input_sizes_,
      PackedTensorAccessor64<scalar_t, 5> q_k_v_,
      const scalar_t inv_sqrt_dim_per_head_,
      const bool assume_aligned_)
      : qkv(qkv_),
        qkv_bias(qkv_bias_),
        offsets(offsets_),
        input_sizes(input_sizes_),
        q_k_v(q_k_v_),
        inv_sqrt_dim_per_head(inv_sqrt_dim_per_head_),
        assume_aligned(assume_aligned_) {}

 private:
  // [B, T, 3 * D], but it's a NestedTensor buffer
  const PackedTensorAccessor64<scalar_t, 1> qkv;
  // [3 * D]
  const PackedTensorAccessor64<scalar_t, 1> qkv_bias;
  const int* offsets;
  const int* input_sizes;
  // [3, B, NH, T, DH]
  PackedTensorAccessor64<scalar_t, 5> q_k_v;
  const scalar_t inv_sqrt_dim_per_head;
  const bool assume_aligned;
};

Tensor collapse_dims_1_and_2(const Tensor& sizes) {
  auto sizes_dim1 = at::native::narrow_symint(sizes, 1, 0, 1);
  auto sizes_dim2 = at::native::narrow_symint(sizes, 1, 1, 1);

  return (sizes_dim1 * sizes_dim2).contiguous();
}

Tensor NestedTensor_batch_offsets_from_size_tensor(
    const Tensor& sizes,
    int64_t extra_elements) {
  int64_t* const sizes_ptr = sizes.data_ptr<int64_t>();
  Tensor offsets = at::empty({1 + sizes.size(0) + extra_elements}, at::kInt);
  int32_t* const offsets_ptr = offsets.mutable_data_ptr<int32_t>();
  offsets_ptr[0] = 0;
  const auto sizes_size_1 = sizes.size(1);
  const auto sizes_size_0 = sizes.size(0);
  for (const auto i : c10::irange(sizes_size_0)) {
    int64_t prod = 1;
    for (const auto j : c10::irange(sizes_size_1)) {
      prod *= sizes_ptr[i * sizes_size_1 + j];
    }
    offsets_ptr[i + 1] = offsets_ptr[i] + prod;
  }
  return offsets;
}

// each nd_range is dealing with one qkv tensor, each nd_group is dealing with D
// dim
template <typename scalar_t, typename accscalar_t>
struct transform_bias_rescale_qkv_kernel_functor {
  void operator()(sycl::nd_item<1> item) const {
    auto NH = q_k_v.size(2);
    auto T = q_k_v.size(3);
    auto DH = q_k_v.size(4);

    const auto qkv_stride_0 = q_k_v.stride(0);
    const auto qkv_stride_1 = q_k_v.stride(1);
    const auto qkv_stride_2 = q_k_v.stride(2);
    const auto qkv_stride_3 = q_k_v.stride(3);
    const auto qkv_stride_4 = q_k_v.stride(4);
    scalar_t* const qkv_data = q_k_v.data();

    const auto group_id = item.get_group(0);
    const auto local_id = item.get_local_id(0);
    const auto local_range = item.get_local_range(0);

    auto t = group_id % T;
    auto b = group_id / T;

    const auto D = NH * DH;

    if (assume_aligned) {
      constexpr int VEC = TRANSFORM_BIAS_RESCALE_VEC;
      using LoadT = native::Memory::aligned_vector_loop<scalar_t, VEC>;
      // here is aligned, no more need ceiling for D / VEC
      for (int32_t d_v = local_id; d_v < D / VEC; d_v += local_range) {
        auto d = d_v * VEC;
        auto nh = d / DH;
        auto dh = d % DH;
        scalar_t qkv_bias_q[VEC];
        scalar_t qkv_bias_k[VEC];
        scalar_t qkv_bias_v[VEC];
        scalar_t qkv_q[VEC];
        scalar_t qkv_k[VEC];
        scalar_t qkv_v[VEC];

        *reinterpret_cast<LoadT*>(&qkv_bias_q) =
            *reinterpret_cast<const LoadT*>(&qkv_bias[d + 0 * D]);
        *reinterpret_cast<LoadT*>(&qkv_bias_k) =
            *reinterpret_cast<const LoadT*>(&qkv_bias[d + 1 * D]);
        *reinterpret_cast<LoadT*>(&qkv_bias_v) =
            *reinterpret_cast<const LoadT*>(&qkv_bias[d + 2 * D]);

        *reinterpret_cast<LoadT*>(&qkv_q) =
            *reinterpret_cast<const LoadT*>(&qkv[b][t][d + 0 * D]);
        *reinterpret_cast<LoadT*>(&qkv_k) =
            *reinterpret_cast<const LoadT*>(&qkv[b][t][d + 1 * D]);
        *reinterpret_cast<LoadT*>(&qkv_v) =
            *reinterpret_cast<const LoadT*>(&qkv[b][t][d + 2 * D]);

#pragma unroll
        for (auto ii = 0; ii < VEC; ii++) {
          qkv_q[ii] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_q[ii]) +
               static_cast<accscalar_t>(qkv_bias_q[ii])) *
              static_cast<accscalar_t>(inv_sqrt_dim_per_head));
          qkv_k[ii] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_k[ii]) +
               static_cast<accscalar_t>(qkv_bias_k[ii])));
          qkv_v[ii] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_v[ii]) +
               static_cast<accscalar_t>(qkv_bias_v[ii])));
        }

        // Here we require DH % VEC == 0 for these vectorized stores.
        *reinterpret_cast<LoadT*>(
            &qkv_data
                [0 * qkv_stride_0 + b * qkv_stride_1 + nh * qkv_stride_2 +
                 t * qkv_stride_3 + dh * qkv_stride_4]) =
            *reinterpret_cast<const LoadT*>(&qkv_q);
        *reinterpret_cast<LoadT*>(
            &qkv_data
                [1 * qkv_stride_0 + b * qkv_stride_1 + nh * qkv_stride_2 +
                 t * qkv_stride_3 + dh * qkv_stride_4]) =
            *reinterpret_cast<const LoadT*>(&qkv_k);
        *reinterpret_cast<LoadT*>(
            &qkv_data
                [2 * qkv_stride_0 + b * qkv_stride_1 + nh * qkv_stride_2 +
                 t * qkv_stride_3 + dh * qkv_stride_4]) =
            *reinterpret_cast<const LoadT*>(&qkv_v);
      }
    } else {
      // without vectorize load and store
      for (int32_t d = local_id; d < D; d += local_range) {
        auto nh = d / DH;
        auto dh = d % DH;
        scalar_t qkv_bias_q = qkv_bias[0 * D + d];
        scalar_t qkv_bias_k = qkv_bias[1 * D + d];
        scalar_t qkv_bias_v = qkv_bias[2 * D + d];
        scalar_t qkv_q = qkv[b][t][0 * D + d];
        scalar_t qkv_k = qkv[b][t][1 * D + d];
        scalar_t qkv_v = qkv[b][t][2 * D + d];

        qkv_q = static_cast<scalar_t>(
            (static_cast<accscalar_t>(qkv_q) +
             static_cast<accscalar_t>(qkv_bias_q)) *
            static_cast<accscalar_t>(inv_sqrt_dim_per_head));
        qkv_k = static_cast<scalar_t>(
            static_cast<accscalar_t>(qkv_k) +
            static_cast<accscalar_t>(qkv_bias_k));
        qkv_v = static_cast<scalar_t>(
            static_cast<accscalar_t>(qkv_v) +
            static_cast<accscalar_t>(qkv_bias_v));

        qkv_data
            [0 * qkv_stride_0 + b * qkv_stride_1 + nh * qkv_stride_2 +
             t * qkv_stride_3 + dh * qkv_stride_4] = qkv_q;
        qkv_data
            [1 * qkv_stride_0 + b * qkv_stride_1 + nh * qkv_stride_2 +
             t * qkv_stride_3 + dh * qkv_stride_4] = qkv_k;
        qkv_data
            [2 * qkv_stride_0 + b * qkv_stride_1 + nh * qkv_stride_2 +
             t * qkv_stride_3 + dh * qkv_stride_4] = qkv_v;
      }
    }
  }

  transform_bias_rescale_qkv_kernel_functor(
      const PackedTensorAccessor64<scalar_t, 3> qkv_,
      const PackedTensorAccessor64<scalar_t, 1> qkv_bias_,
      PackedTensorAccessor64<scalar_t, 5> q_k_v_,
      const scalar_t inv_sqrt_dim_per_head_,
      const bool assume_aligned_)
      : qkv(qkv_),
        qkv_bias(qkv_bias_),
        q_k_v(q_k_v_),
        inv_sqrt_dim_per_head(inv_sqrt_dim_per_head_),
        assume_aligned(assume_aligned_) {}

 private:
  // [B, T, 3 * D]
  const PackedTensorAccessor64<scalar_t, 3> qkv;
  // [3 * D]
  const PackedTensorAccessor64<scalar_t, 1> qkv_bias;
  // [3, B, num_heads, T, dim_per_head]
  PackedTensorAccessor64<scalar_t, 5> q_k_v;
  const scalar_t inv_sqrt_dim_per_head;
  const bool assume_aligned;
};

} // namespace impl

int64_t _fused_sdp_choice(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale) {
  // We have implemented efficient_attention backend with xetla, flash_attention
  // backend is not supported now, which will be implemented in the future. So
  // we provide two backends here.
  sdp::sdp_params kernel_params{
      query, key, value, attn_mask_, dropout_p, is_causal};
  sdp::SDPBackend backend = sdp::use_mem_efficient_attention(kernel_params)
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

// compute q = (q + q_bias) / sqrt(dim_per_head), k = k + k_bias, v = v + v_bias
// Note: current only support contiguous indexing, since nested tensor is all
// contiguous
std::tuple<Tensor, Tensor, Tensor> _transform_bias_rescale_qkv(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    const int64_t num_head) {
  // for nested tensor, B is most outer size, but T is not regular, it should be
  // the large size on dim1
  auto B = qkv.is_nested()
      ? at::native::get_nested_tensor_impl(qkv)->get_nested_sizes().size(0)
      : qkv.size(0);

  auto T = qkv.is_nested() ? at::native::NestedTensor_get_max_size(
                                 *at::native::get_nested_tensor_impl(qkv))[0]
                           : qkv.size(1);

  // qkv_bias size should be same with finall linear projection layer, which
  // size is 3 * D
  auto _3D = qkv_bias.size(0);
  auto D = _3D / 3;
  TORCH_CHECK(D % num_head == 0);
  const auto dim_per_head = D / num_head;

  // q_k_v B T 3D -> 3, B, num_head, T, dim_per_head
  auto q_k_v = at::empty({3, B, num_head, T, dim_per_head}, qkv_bias.options());

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto max_wg_size = dpcppMaxWorkGroupSize();

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      qkv.scalar_type(),
      "transform_bias_rescale_qkv",
      [&] {
        using accscalar_t = at::AtenIpexTypeXPU::acc_type<scalar_t>;
        auto local_range = std::max(
            std::min<int32_t>(
                max_wg_size,
                (D + TRANSFORM_BIAS_RESCALE_VEC - 1) /
                    TRANSFORM_BIAS_RESCALE_VEC),
            1);
        auto global_range = B * T;
        const bool aligned =
            ((dim_per_head % TRANSFORM_BIAS_RESCALE_VEC) == 0) &&
            ((reinterpret_cast<intptr_t>(qkv_bias.data_ptr()) %
              TRANSFORM_BIAS_RESCALE_VEC) == 0);

        if (aligned) {
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
              D % TRANSFORM_BIAS_RESCALE_VEC == 0,
              "D = num_heads * dim_per_head, so we should have dim_per_head % "
              "TRANSFORM_BIAS_RESCALE_VEC == 0 => "
              "D % TRANSFORM_BIAS_RESCALE_VEC == 0");
        }

        if (qkv.is_nested()) {
          auto* nt_qkv = at::native::get_nested_tensor_impl(qkv);
          const at::Tensor& nt_qkv_buffer = nt_qkv->get_buffer();

          auto sizes = impl::collapse_dims_1_and_2(nt_qkv->get_nested_sizes());
          auto offsets = impl::NestedTensor_batch_offsets_from_size_tensor(
              sizes, sizes.numel());

          at::native::narrow_symint(
              offsets, 0, sizes.numel() + 1, sizes.numel())
              .copy_(sizes.reshape({-1}));
          auto metadata = offsets.to(at::Device(kXPU), at::kInt, true, true);
          const auto offsets_ptr = metadata.data_ptr<int>();
          const auto sizes_ptr = offsets_ptr + sizes.numel() + 1;
          const auto input_dim = sizes.sizes()[1];
          auto qkv_acc = q_k_v.packed_accessor64<scalar_t, 5>();
          if (aligned &&
              ((reinterpret_cast<intptr_t>(qkv.data_ptr()) %
                TRANSFORM_BIAS_RESCALE_VEC) == 0)) {
            auto cgf = DPCPP_Q_CGF(cgh) {
              impl::transform_bias_rescale_qkv_add_padding_impl<
                  scalar_t,
                  accscalar_t>
                  kfn(nt_qkv_buffer.packed_accessor64<scalar_t, 1>(),
                      qkv_bias.packed_accessor64<scalar_t, 1>(),
                      offsets_ptr,
                      sizes_ptr,
                      qkv_acc,
                      1.0 / std::sqrt(static_cast<scalar_t>(dim_per_head)),
                      true);

              cgh.parallel_for<decltype(kfn)>(
                  sycl::nd_range<1>(
                      sycl::range<1>(global_range * local_range),
                      sycl::range<1>(local_range)),
                  kfn);
            };
            DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
          } else {
            auto cgf = DPCPP_Q_CGF(cgh) {
              impl::transform_bias_rescale_qkv_add_padding_impl<
                  scalar_t,
                  accscalar_t>
                  kfn(nt_qkv_buffer.packed_accessor64<scalar_t, 1>(),
                      qkv_bias.packed_accessor64<scalar_t, 1>(),
                      offsets_ptr,
                      sizes_ptr,
                      qkv_acc,
                      1.0 / std::sqrt(static_cast<scalar_t>(dim_per_head)),
                      false);

              cgh.parallel_for<decltype(kfn)>(
                  sycl::nd_range<1>(
                      sycl::range<1>(global_range * local_range),
                      sycl::range<1>(local_range)),
                  kfn);
            };
            DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
          }
        } else {
          auto cgf = DPCPP_Q_CGF(cgh) {
            impl::
                transform_bias_rescale_qkv_kernel_functor<scalar_t, accscalar_t>
                    kfn(qkv.packed_accessor64<scalar_t, 3>(),
                        qkv_bias.packed_accessor64<scalar_t, 1>(),
                        q_k_v.packed_accessor64<scalar_t, 5>(),
                        1.0 / std::sqrt(static_cast<scalar_t>(dim_per_head)),
                        aligned);

            cgh.parallel_for<decltype(kfn)>(
                sycl::nd_range<1>(
                    sycl::range<1>(global_range * local_range),
                    sycl::range<1>(local_range)),
                kfn);
          };
          DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
        }
      });

  auto q_k_v_s =
      at::native::split(q_k_v.view({3 * B, num_head, T, dim_per_head}), B, 0);
  return std::make_tuple(q_k_v_s[0], q_k_v_s[1], q_k_v_s[2]);
}

bool check_for_seq_len_1_nested_tensor(sdp::sdp_params params, bool debug) {
  // When this function is called we are assured that the nt is dim==4
  if (!params.query.is_nested()) {
    return true;
  }

  const auto nt_q_tensor_impl =
      at::native::get_nested_tensor_impl(params.query);
  const at::Tensor& sizes = nt_q_tensor_impl->get_nested_sizes();
  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const int64_t n_tensors = params.query.size(0);
  const int64_t size_tensor_stride = sizes.stride(0);

  // This is being called inside sdp with shape [batch, heads, {seq_len}, dim]
  for (const auto i : c10::irange(n_tensors)) {
    if (sizes_ptr[(i * size_tensor_stride) + 1] <= 1) {
      if (debug) {
        TORCH_WARN(
            "Packed projection for fused kernels does not support sequence_length <= 1");
      }
      return false;
    }
  }

  return true;
}

// B for batch size
// T for seq length
// D for embed_dimension
std::tuple<Tensor, Tensor> _native_multi_head_attention(
    // query shape: [B, T, D]
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const int64_t embed_dim,
    const int64_t num_head,
    // qkv_weight shape: [3 * D, D]
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const c10::optional<Tensor>& mask,
    bool need_weights,
    bool average_attn_weights,
    const c10::optional<int64_t> mask_type) {
  TORCH_CHECK(
      !mask || !query.is_nested(),
      "NestedTensor with mask is not supported yet");
  const auto D = embed_dim;
  TORCH_CHECK(
      query.dim() == 3, "expected 3-D `query`, got ", query.dim(), "-D tensor");
  TORCH_CHECK(
      query.is_nested() || query.sizes()[2] == embed_dim,
      "passed-in embed_dim ",
      embed_dim,
      " didn't match last dim of query ",
      query.sizes()[2]);
  TORCH_CHECK(
      key.dim() == 3, "expected 3-D `key`, got ", key.dim(), "-D tensor");
  TORCH_CHECK(
      value.dim() == 3, "expected 3-D `value`, got ", value.dim(), "-D tensor");
  TORCH_CHECK(
      query.is_nested() || key.is_nested() || value.is_nested() ||
          (query.sizes() == key.sizes() && key.sizes() == value.sizes()),
      "expected `query`/`key`/`value` shapes to match");
  TORCH_CHECK(
      qkv_weight.dim() == 2,
      "expected 2-D `qkv_weight`, got ",
      qkv_weight.dim(),
      "-D tensor");
  TORCH_CHECK(
      D * 3 == qkv_weight.sizes()[0],
      "expected `qkv_weight` first dim to be 3x embed_dim");
  TORCH_CHECK(
      D == qkv_weight.sizes()[1],
      "expected `qkv_weight` second dim to be embed_Dim");
  TORCH_CHECK(
      qkv_bias.dim() == 1,
      "expected 1-D `qkv_bias`, got ",
      qkv_bias.dim(),
      "-D tensor");
  TORCH_CHECK(
      qkv_bias.sizes()[0] == 3 * D,
      "expected `qkv_bias` first dim and first dim of query to be equal");
  TORCH_CHECK(
      D % num_head == 0, "`embed_dim` must divide evenly by `num_heads`");

  const auto B = query.is_nested()
      ? native::get_nested_tensor_impl(query)->get_nested_sizes().size(0)
      : query.sizes()[0];
  auto T = query.is_nested() ? 0 : query.sizes()[1];
  const auto dim_per_head = D / num_head;

  if ((query.is_same(key) && key.is_same(value)) && dim_per_head % 8 == 0 &&
      !need_weights) {
    // We have not done linear projection yet but the input for SDP
    // Is expected to be 4 dimensional. We "cheaply" create view tensors
    // That will then be used for checking hot path conditions with
    // select_sd_backend
    auto q =
        query.view({query.size(0), -1, num_head, dim_per_head}).transpose(1, 2);
    auto k =
        key.view({key.size(0), -1, num_head, dim_per_head}).transpose(1, 2);
    auto v =
        value.view({value.size(0), -1, num_head, dim_per_head}).transpose(1, 2);

    auto backend = static_cast<sdp::SDPBackend>(
        _fused_sdp_choice(q, k, v, mask, 0.0, false));
    sdp::sdp_params kernel_params{q, k, v, mask, 0.0, false};

    bool no_seq_len_1_nested = query.is_nested()
        ? check_for_seq_len_1_nested_tensor(kernel_params, false)
        : true;
    if (!mask.has_value() && no_seq_len_1_nested &&
        (backend == sdp::SDPBackend::flash_attention ||
         backend == sdp::SDPBackend::efficient_attention)) {
      auto x = at::linear(query, qkv_weight, qkv_bias);
      auto chunks = x.chunk(3, -1);
      auto x_size_0 = x.size(0);

      chunks[0] = (chunks[0].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      chunks[1] = (chunks[1].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      chunks[2] = (chunks[2].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);

      auto y = at::scaled_dot_product_attention(
          chunks[0], chunks[1], chunks[2], mask, 0.0, false, c10::nullopt);

      auto past_sdp = y.transpose(1, 2).reshape({x_size_0, -1, embed_dim});
      return std::make_tuple(
          at::linear(past_sdp, proj_weight, proj_bias), Tensor());
    }
  }

  auto qkv = native::qkv_projection(query, key, value, embed_dim, qkv_weight);

  if (!qkv.is_nested() && qkv.numel() == 0) {
    if (query.is_nested()) {
      return std::make_tuple(Tensor(), Tensor());
    }
    return std::make_tuple(at::empty_like(query), Tensor());
  }

  // for non-nested tensor, the size is ir-regular, so just get the size(1) is
  // legal
  if (!query.is_nested() || !qkv.is_nested()) {
    if (query.is_nested()) {
      T = qkv.size(1);
    }
    native::debug_assert_shape(__LINE__, qkv, {B, T, 3 * D});
  }

  auto q_k_v = AtenIpexTypeNestedTensorXPU::_transform_bias_rescale_qkv(
      qkv, qkv_bias, num_head);
  qkv = Tensor(); // Not used any more, allow free
  auto& q = std::get<0>(q_k_v);
  const auto& k = std::get<1>(q_k_v);
  const auto& v = std::get<2>(q_k_v);
  native::debug_assert_shape(__LINE__, q, {B, num_head, T, dim_per_head});
  native::debug_assert_shape(__LINE__, k, {B, num_head, T, dim_per_head});
  native::debug_assert_shape(__LINE__, v, {B, num_head, T, dim_per_head});

  // shape: [B, num_head, T, T]
  auto qkt = native::bmm_nt(q, k);
  // q & k are dead but cannot be freed because they were packed with v
  native::debug_assert_shape(__LINE__, qkt, {B, num_head, T, T});

  // shape: [B, num_head, T, T]
  // TODO: long-term, have a kernel that works with
  // NestedTensor directly if there is no mask passed
  qkt = native::masked_softmax(qkt, mask, query, mask_type);

  // shape: [B, num_head, T, dim_per_head]
  // reuse storage for q; we're done with it
  auto attn_ctx = native::bmm_nn(q, qkt, v);
  // qkv is not dead; we just reused storage for q!
  if (!need_weights) {
    qkt = Tensor();
  }

  native::debug_assert_shape(
      __LINE__, attn_ctx, {B, num_head, T, dim_per_head});

  // shape: [B, T, D]
  // Fuse transform_0213 inside
  auto proj = native::transform0213_gemm_nt_bias(
      attn_ctx, proj_weight, proj_bias, query);
  native::debug_assert_shape(__LINE__, proj, {B, T, D});

  if (need_weights && average_attn_weights) {
    // weights are not needed for full transformer, so don't worry too
    // much about performance -- we implement this just to make use
    // cases that don't disable need_weights still get some speedup.
    qkt = qkt.sum(1);
    qkt /= num_head;
  }

  return std::make_tuple(std::move(proj), std::move(qkt));
}
} // namespace AtenIpexTypeNestedTensorXPU

namespace AtenIpexTypeXPU {
std::tuple<Tensor, Tensor, Tensor> _transform_bias_rescale_qkv(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    int64_t num_head) {
  return at::AtenIpexTypeNestedTensorXPU::_transform_bias_rescale_qkv(
      qkv, qkv_bias, num_head);
}

std::tuple<Tensor, Tensor> _native_multi_head_attention(
    // query shape: [B, T, D]
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const int64_t embed_dim,
    const int64_t num_head,
    // qkv_weight shape: [3 * D, D]
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const c10::optional<Tensor>& mask,
    bool need_weights,
    bool average_attn_weights,
    const c10::optional<int64_t> mask_type) {
  return at::AtenIpexTypeNestedTensorXPU::_native_multi_head_attention(
      query,
      key,
      value,
      embed_dim,
      num_head,
      qkv_weight,
      qkv_bias,
      proj_weight,
      proj_bias,
      mask,
      need_weights,
      average_attn_weights,
      mask_type);
}

} // namespace AtenIpexTypeXPU

} // namespace at

namespace {

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "xetla_paged_attention_v1.xpu",
      at::AtenIpexTypeXPU::xetla_paged_attention_v1,
      c10::DispatchKey::XPU);
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "xetla_paged_attention_v2.xpu",
      at::AtenIpexTypeXPU::xetla_paged_attention_v2,
      c10::DispatchKey::XPU);
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "_transform_bias_rescale_qkv",
      at::AtenIpexTypeXPU::_transform_bias_rescale_qkv,
      c10::DispatchKey::XPU);
}

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "_native_multi_head_attention",
      at::AtenIpexTypeXPU::_native_multi_head_attention,
      c10::DispatchKey::XPU);
}

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

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "xetla_sdp_dropout",
      at::AtenIpexTypeXPU::xetla_sdp_dropout,
      c10::DispatchKey::AutogradXPU);
}
} // namespace
