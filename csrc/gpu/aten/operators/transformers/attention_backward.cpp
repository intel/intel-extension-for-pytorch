#include <ATen/ATen.h>
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
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    const bool bias_requires_grad,
    const c10::optional<double> scale) {
#if defined(USE_XETLA)
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

  Tensor grad_q, grad_k, grad_v, grad_bias;
  grad_q = at::zeros_like(query);
  grad_k = at::empty_like(key);
  grad_v = at::empty_like(value);

  // TODO: broadcast support
  uint32_t attn_mask_padded_block_size = 0;
  Tensor attn_mask_c;
  if (bias.has_value()) {
    TORCH_CHECK(
        bias.value().stride(1) == 0 || bias.value().size(1) == 1,
        "XPU efficient attention requires attn mask second dim is 1");

    attn_mask_c = bias.value();
    attn_mask_padded_block_size = bias->stride(-2);
    // broadcast attn_mask at the 3nd dim
    if (bias.value().stride(-2) == 0) {
      // attn_mask_c shape [batchsize, 1, q_len, k_len]
      attn_mask_c = bias.value()
                        .as_strided(
                            {bias.value().size(0),
                             1,
                             bias.value().size(2),
                             bias.value().size(3)},
                            bias.value().strides())
                        .contiguous();
      // padding will be removed after contiguous
      if (attn_mask_c.stride(-2) % 16 != 0) {
        attn_mask_c = sdp::pad_bias<16>(attn_mask_c);
      }
      attn_mask_padded_block_size = attn_mask_c.size(-1);
    }

    if (bias_requires_grad) {
      // force alignment for the last dim
      std::vector<int64_t> sz = bias->sizes().vec();
      int64_t lastDim = sz[sz.size() - 1];
      sz[sz.size() - 1] = attn_mask_padded_block_size;
      // grad_bias shape [batchsize, head_size, q_len, k_len]
      //          layout [batchsize, head_size, q_len, padding_size]
      grad_bias = at::empty(sz, bias->options())
                      .slice(/*dim=*/-1, /*start=*/0, /*end=*/lastDim);
    }
  }

  Tensor workspace = at::empty_like(logsumexp);
  const double softmax_scale =
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(query.size(-1)));

  const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;
  XetlaType xeType = sdp::aten_to_Xetla_dtype(query);
  auto dpcpp_queue = dpcppGetCurrentQueue();

  fmha_backward_kernel(
      xeType,
      dpcpp_queue,
      grad_out.data_ptr(),
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      bias.has_value() ? attn_mask_c.data_ptr() : (void*)nullptr,
      out.data_ptr(),
      logsumexp.data_ptr(),
      workspace.data_ptr(),
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
      attn_mask_padded_block_size,
      is_causal,
      use_dropout,
      (uint64_t)*philox_seed.data_ptr<int64_t>(),
      (uint64_t)*philox_offset.data_ptr<int64_t>());
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
      philox_seed,
      philox_offset,
      grad_input_mask[3],
      scale);
}

} // namespace AtenIpexTypeXPU
} // namespace at
