#pragma once
#include <ATen/Tensor.h>
namespace torch_ipex {
namespace cpu {
namespace bf16 {
namespace converter {

void bf16_to_fp32(void *dst, const void *src, int len);
void fp32_to_bf16(void *dst, const void *src, int len);
at::Tensor cat_bfloat16_float(const at::Tensor top_half, const at::Tensor bottom_half);
std::tuple<at::Tensor, at::Tensor> split_float_bfloat16(const at::Tensor tensor);
}  // namespace converter
}  // namespace bf16
}  // namespace cpu
}  // namespace torch_ipex
