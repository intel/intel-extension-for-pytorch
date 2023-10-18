#pragma once

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <c10/util/SmallVector.h>
#include <immintrin.h>
#include <limits>
#include "utils.h"

namespace torch_ipex {
namespace cpu {
namespace kernel {
template <typename T, typename T1>
void _compute_rmsnorm(
    const T* a_ptr,
    const int& size,
    float eps,
    const T1* gamma_ptr,
    T* out_ptr) {
  auto vec_acc_pow = _mm512_set1_ps(0.0);
  int i;
  for (i = 0; i <= size - 16; i += 16) {
    auto vec_a = _loadu(a_ptr + i);
    auto s = vec_a * vec_a;
    vec_acc_pow += s;
  }
  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    auto vec_a = _maskz_loadu(a_ptr + i, mask);
    auto s = vec_a * vec_a;
    vec_acc_pow += s;
  }
  float var_val = _mm512_reduce_add_ps(vec_acc_pow) / static_cast<float>(size);
  float scale = float(1.0) / std::sqrt(var_val + eps);
  auto vec_scale = _mm512_set1_ps(scale);
  for (i = 0; i <= size - 16; i += 16) {
    auto vec_input = _loadu(a_ptr + i);
    auto vec_gamma = _mm512_set1_ps(1.0);
    if (gamma_ptr) {
      vec_gamma = _loadu(gamma_ptr + i);
    }
    auto vec_res = vec_input * vec_scale * vec_gamma;
    _storeu(out_ptr + i, vec_res);
  }
  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    auto vec_input = _maskz_loadu(a_ptr + i, mask);
    auto vec_gamma = _mm512_set1_ps(1.0);
    if (gamma_ptr) {
      vec_gamma = _maskz_loadu(gamma_ptr + i, mask);
    }
    auto vec_res = vec_input * vec_scale * vec_gamma;
    _mask_storeu(out_ptr + i, vec_res, mask);
  }
}

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
