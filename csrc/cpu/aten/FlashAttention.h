#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

namespace {

std::tuple<at::Tensor, at::Tensor> flash_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    c10::optional<at::Tensor> attention_mask,
    c10::optional<double> scale);
} // namespace

using flash_attention_kernel_fn = std::tuple<at::Tensor, at::Tensor> (*)(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    c10::optional<at::Tensor> attention_mask,
    c10::optional<double> scale);

IPEX_DECLARE_DISPATCH(flash_attention_kernel_fn, flash_attention_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
