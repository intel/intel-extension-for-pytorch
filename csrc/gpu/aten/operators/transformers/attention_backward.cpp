#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <runtime/Device.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "sdp_utils.h"

namespace at {
namespace AtenIpexTypeXPU {

std::tuple<Tensor, Tensor, Tensor, Tensor> _efficient_attention_backward_impl(
    const Tensor& _grad_out,
    const Tensor& _query,
    const Tensor& _key,
    const Tensor& _value,
    const c10::optional<at::Tensor>& bias,
    const Tensor& _out,
    const Tensor& logsumexp,
    bool is_causal,
    double dropout_p,
    const c10::optional<at::Tensor>& dropout_mask,
    const c10::optional<at::Tensor>& philox_seed,
    const c10::optional<at::Tensor>& philox_offset,
    const bool bias_requires_grad,
    const c10::optional<double> scale) {
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)
  if (!_grad_out.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
  }

  TORCH_CHECK(
      dpcppGetDeviceHasXMX(),
      "SDP backward kernel requires XMX, but the current platform has no XMX ...");

  auto grad_out = _grad_out.transpose(1, 2).contiguous().transpose(1, 2);
  auto query = _query.transpose(1, 2).contiguous().transpose(1, 2);
  auto key = _key.transpose(1, 2).contiguous().transpose(1, 2);
  auto value = _value.transpose(1, 2).contiguous().transpose(1, 2);
  auto out = _out.transpose(1, 2).contiguous().transpose(1, 2);

  Tensor grad_q_tmp, grad_q, grad_k, grad_v, grad_bias;
  // tmp grad_q for accumulation
  grad_q_tmp = at::zeros(query.sizes(), query.options().dtype(kFloat));
  grad_q = at::empty_like(query);
  grad_k = at::empty_like(key);
  grad_v = at::empty_like(value);

  uint32_t attn_mask_padded_block_size = 0;
  if (bias.has_value()) {
    std::vector<int64_t> sz = bias->sizes().vec();
    int64_t lastDim = sz[sz.size() - 1];
    int64_t alignTo = 16;
    attn_mask_padded_block_size = alignTo * ((lastDim + alignTo - 1) / alignTo);
  }

  if (bias_requires_grad) {
    // force alignment for the last dim
    std::vector<int64_t> sz = bias->sizes().vec();
    int64_t lastDim = sz[sz.size() - 1];
    int64_t alignTo = 16;
    sz[sz.size() - 1] = alignTo * ((lastDim + alignTo - 1) / alignTo);
    // grad_bias shape [batchsize, head_size, q_len, k_len]
    //          layout [batchsize, head_size, q_len, padding_size]
    grad_bias = at::empty(sz, bias->options())
                    .slice(/*dim=*/-1, /*start=*/0, /*end=*/lastDim);
  }

  Tensor workspace = at::empty_like(logsumexp);
  const double softmax_scale =
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(query.size(-1)));

  const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;
  auto xeType = sdp::aten_to_Xetla_dtype(query);
  auto dpcpp_queue = dpcppGetCurrentQueue();

  auto cgfs = fmha_backward_kernel(
      xeType,
      grad_out.data_ptr(),
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      bias.has_value() ? bias.value().data_ptr() : (void*)nullptr,
      dropout_mask.has_value() ? dropout_mask.value().data_ptr()
                               : (void*)nullptr,
      out.data_ptr(),
      logsumexp.data_ptr(),
      workspace.data_ptr(),
      grad_q_tmp.data_ptr(),
      softmax_scale,
      dropout_p,
      grad_q.data_ptr(),
      grad_k.data_ptr(),
      grad_v.data_ptr(),
      bias_requires_grad ? grad_bias.data_ptr() : (void*)nullptr,
      query.size(0),
      query.size(1),
      query.size(3),
      query.size(2),
      key.size(2),
      bias.has_value() ? bias->stride(0) : -1,
      bias.has_value() ? bias->stride(1) : -1,
      bias.has_value() ? bias->stride(2) : -1,
      attn_mask_padded_block_size,
      is_causal,
      use_dropout,
      philox_seed.has_value()
          ? (uint64_t)*philox_seed.value().data_ptr<int64_t>()
          : -1,
      philox_offset.has_value()
          ? (uint64_t)*philox_offset.value().data_ptr<int64_t>()
          : -1);
  DPCPP_Q_SUBMIT_CGFS(dpcpp_queue, cgfs);
  return std::make_tuple(grad_q, grad_k, grad_v, grad_bias);
#else
  AT_ERROR("SDP backward: xetla library not found in compilation");
  return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
#endif
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
_scaled_dot_product_efficient_attention_backward(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_bias,
    const Tensor& out,
    const Tensor& logsumexp,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    double dropout_p,
    std::array<bool, 4> grad_input_mask,
    bool causal,
    c10::optional<double> scale) {
  if (!grad_out.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
  }

  // This is needed because SaveVarible automatically converts
  // c10::optional to undefined tensor
  c10::optional<Tensor> kernel_bias;
  if (attn_bias.defined()) {
    kernel_bias = attn_bias;
  }

  return _efficient_attention_backward_impl(
      grad_out,
      query,
      key,
      value,
      kernel_bias,
      out,
      logsumexp,
      causal,
      dropout_p,
      c10::nullopt,
      philox_seed,
      philox_offset,
      grad_input_mask[3],
      scale);
}

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
    c10::optional<double> scale) {
  RECORD_FUNCTION("ipex_sdp_dropout_backward", {});
  if (!grad_out.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
  }

  return _efficient_attention_backward_impl(
      grad_out,
      query,
      key,
      value,
      attn_bias,
      out,
      logsumexp,
      causal,
      dropout_p,
      dropout_mask,
      c10::nullopt,
      c10::nullopt,
      grad_input_mask,
      scale);
}

} // namespace AtenIpexTypeXPU
} // namespace at
