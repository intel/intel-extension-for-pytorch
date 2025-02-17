#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

namespace {

at::Tensor decode_attention_forward_cpu(
    at::Tensor& query,
    at::Tensor& output,
    at::Tensor& kv_cache,
    at::Tensor& beam_idx,
    at::Tensor& attn_logits,
    double scaling,
    double logit_cap,
    int64_t offset);
} // namespace

using decode_attention_kernel_fn = at::Tensor (*)(
    at::Tensor& query,
    at::Tensor& output,
    at::Tensor& kv_cache,
    at::Tensor& beam_idx,
    at::Tensor& attn_logits,
    const double scaling,
    const double logit_cap,
    int64_t offset);
IPEX_DECLARE_DISPATCH(decode_attention_kernel_fn, decode_attention_kernel_stub);

} // namespace cpu
} // namespace torch_ipex