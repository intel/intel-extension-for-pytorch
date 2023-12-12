#pragma once

#include <ATen/ATen.h>
#include <ATen/cpu/vec/vec.h>
#include <torch/types.h>

namespace torch_ipex {
namespace cpu {
namespace kernel {

using namespace at::vec;

template <typename scalar_t>
inline void apply_rope_along_head_kernel(
    scalar_t* in_ptr_start,
    scalar_t* out_ptr_start,
    float* cos_start,
    float* sin_start,
    int64_t rotary_ndims,
    int64_t offset) {
  auto h = 0;
  for (h = 0; h < rotary_ndims / 2; h++) {
    float x = in_ptr_start[h];
    float y = in_ptr_start[h + offset];
    float sin = sin_start[h];
    float cos = cos_start[h];
    float out0 = x * cos - y * sin;
    float out1 = y * cos + x * sin;
    out_ptr_start[h] = out0;
    out_ptr_start[h + offset] = out1;
  }
}

template <>
inline void apply_rope_along_head_kernel(
    float* in_ptr_start,
    float* out_ptr_start,
    float* cos_start,
    float* sin_start,
    int64_t rotary_ndims,
    int64_t offset) {
  auto h = 0;
  using Vec = Vectorized<float>;
  const int vec_size = Vec::size();
  for (h = 0; h <= rotary_ndims / 2 - vec_size; h += vec_size) {
    auto x = Vec::loadu(in_ptr_start + h);
    auto y = Vec::loadu(in_ptr_start + h + offset);
    auto sin = Vec::loadu(sin_start + h);
    auto cos = Vec::loadu(cos_start + h);
    auto out0 = x * cos - y * sin;
    auto out1 = y * cos + x * sin;
    out0.store(out_ptr_start + h);
    out1.store(out_ptr_start + h + offset);
  }
  for (; h < rotary_ndims / 2; h++) {
    float x = in_ptr_start[h];
    float y = in_ptr_start[h + offset];
    float sin = sin_start[h];
    float cos = cos_start[h];
    float out0 = x * cos - y * sin;
    float out1 = y * cos + x * sin;
    out_ptr_start[h] = out0;
    out_ptr_start[h + offset] = out1;
  }
}

template <>
inline void apply_rope_along_head_kernel(
    at::BFloat16* in_ptr_start,
    at::BFloat16* out_ptr_start,
    float* cos_start,
    float* sin_start,
    int64_t rotary_ndims,
    int64_t offset) {
  auto h = 0;
  using bVec = Vectorized<at::BFloat16>;
  using fVec = Vectorized<float>;
  const int fvec_size = fVec::size();
  const int bvec_size = bVec::size();
  for (h = 0; h <= rotary_ndims / 2 - bvec_size; h += bvec_size) {
    bVec x = bVec::loadu(in_ptr_start + h);
    bVec y = bVec::loadu(in_ptr_start + h + offset);
    fVec x0, x1, y0, y1;
    std::tie(x0, x1) = convert_bfloat16_float(x);
    std::tie(y0, y1) = convert_bfloat16_float(y);
    fVec c0 = fVec::loadu(cos_start + h);
    fVec s0 = fVec::loadu(sin_start + h);
    fVec c1 = fVec::loadu(cos_start + h + fvec_size);
    fVec s1 = fVec::loadu(sin_start + h + fvec_size);
    fVec x_out0 = x0 * c0 - y0 * s0;
    fVec x_out1 = x1 * c1 - y1 * s1;
    fVec y_out0 = y0 * c0 + x0 * s0;
    fVec y_out1 = y1 * c1 + x1 * s1;
    bVec x_out = convert_float_bfloat16(x_out0, x_out1);
    bVec y_out = convert_float_bfloat16(y_out0, y_out1);
    x_out.store(out_ptr_start + h);
    y_out.store(out_ptr_start + h + offset);
  }
  for (; h < rotary_ndims / 2; h++) {
    float x = in_ptr_start[h];
    float y = in_ptr_start[h + offset];
    float sin = sin_start[h];
    float cos = cos_start[h];
    float out0 = x * cos - y * sin;
    float out1 = y * cos + x * sin;
    out_ptr_start[h] = out0;
    out_ptr_start[h + offset] = out1;
  }
}
} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
