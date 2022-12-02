#pragma once
#include <ATen/Tensor.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

namespace {

void bf16_to_fp32(void* dst, const void* src, int len);
void fp32_to_bf16(void* dst, const void* src, int len);
at::Tensor cat_bfloat16_float_kernel_impl(
    const at::Tensor top_half,
    const at::Tensor bottom_half);

std::tuple<at::Tensor, at::Tensor> split_float_bfloat16_kernel_impl(
    const at::Tensor tensor);

} // namespace

using cat_bfloat16_float_kernel_fn =
    at::Tensor (*)(const at::Tensor, const at::Tensor);
DECLARE_DISPATCH(cat_bfloat16_float_kernel_fn, cat_bfloat16_float_kernel_stub);

using split_float_bfloat16_kernel_fn =
    std::tuple<at::Tensor, at::Tensor> (*)(const at::Tensor);
DECLARE_DISPATCH(
    split_float_bfloat16_kernel_fn,
    split_float_bfloat16_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
