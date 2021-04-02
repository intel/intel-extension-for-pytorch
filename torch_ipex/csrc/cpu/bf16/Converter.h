#pragma once

namespace torch_ipex {
namespace cpu {
namespace bf16 {
namespace converter {

void bf16_to_fp32(void *dst, const void *src, int len);
void fp32_to_bf16(void *dst, const void *src, int len);

}  // namespace converter
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
