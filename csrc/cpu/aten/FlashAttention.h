#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

namespace {

at::Tensor flash_attention(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    const double scale_attn,
    at::Tensor attention_mask);
}

using flash_attention_kernel_fn = at::Tensor (*)(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    const double scale_attn,
    at::Tensor attention_mask);

DECLARE_DISPATCH(flash_attention_kernel_fn, flash_attention_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
