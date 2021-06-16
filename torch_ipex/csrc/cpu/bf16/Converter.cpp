#include "Converter.h"

#include <ATen/ATen.h>

#if defined(AVX512)
#include "vec/vec_type_cvt.h"
#define BF16_2_FP32(dst, src, len) cvt_bf16_to_fp32(dst, src, len)
#define FP32_2_BF16(dst, src, len) cvt_fp32_to_bf16(dst, src, len)
#else
#define BF16_2_FP32(dst, src, len)
#define FP32_2_BF16(dst, src, len)
#endif

namespace torch_ipex {
namespace cpu {
namespace bf16 {
namespace converter {

void bf16_to_fp32(void *dst, const void *src, int len) {
  BF16_2_FP32((float *)dst, (at::BFloat16 *)src, len);
}

void fp32_to_bf16(void *dst, const void *src, int len) {
  FP32_2_BF16((at::BFloat16 *)dst, (float *)src, len);
}

}  // namespace converter
}  // namespace dbl
}  // namespace cpu
}  // namespace torch_ipex
