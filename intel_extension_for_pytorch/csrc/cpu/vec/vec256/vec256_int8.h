#pragma once
#include <cstdlib>

namespace torch_ipex {
namespace cpu {
namespace kernel {

static inline __attribute__((always_inline)) void scale_and_store_int8(
    int8_t* out,
    const int8_t* in,
    float& scale,
    int64_t len) {
  for (int64_t i = 0; i < len; i++) {
    int32_t i32_val = *(in + i);
    float ps_val = (float)i32_val;
    ps_val *= scale;
    i32_val = int32_t(std::round(ps_val));
    if (i32_val < INT8_MIN) {
      *(out + i) = INT8_MIN;
    } else if (i32_val > INT8_MAX) {
      *(out + i) = INT8_MAX;
    } else {
      *(out + i) = (int8_t)i32_val;
    }
  }
}

static inline void scale_and_move_ker(
    int8_t* out,
    const int8_t* in,
    float scale,
    int64_t len) {
  scale_and_store_int8(out, in, scale, len);
}

static inline __attribute__((always_inline)) int32_t _scale_int32(
    int32_t value,
    float scale) {
  float f_val = float(value) * scale;
  int32_t i32_val = int32_t(std::round(f_val));
  if (i32_val < INT8_MIN) {
    i32_val = INT8_MIN;
  } else if (i32_val > INT8_MAX) {
    i32_val = INT8_MAX;
  }
  return i32_val;
}

static inline __attribute__((always_inline)) int8_t _dot_s8s8_scale_s32s8(
    const int8_t* a,
    const int8_t* b,
    size_t len,
    float scale) {
  int32_t c = 0;
  size_t i = 0;
  for (; i < len; i++) {
    c += (int32_t)a[i] * (int32_t)b[i];
  }
  c = _scale_int32(c, scale);
  return (int8_t)c;
}

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
