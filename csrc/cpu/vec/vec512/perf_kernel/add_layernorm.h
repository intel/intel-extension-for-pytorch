#pragma once

#include <immintrin.h>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <c10/util/SmallVector.h>
#include <limits>
#include "utils.h"

namespace torch_ipex {
namespace cpu {
namespace kernel {

template <typename T>
std::pair<float, float> _add_and_compute_mean_var(
    const T* a_ptr,
    const T* b_ptr,
    const int& size,
    float* out) {
  // compute add and mean/var of the value after add
  // we should firstly store add value
  auto vec_a = _loadu(a_ptr);
  auto vec_b = _loadu(b_ptr);
  auto vec_add = _mm512_add_ps(vec_a, vec_b);
  auto vec_acc_mean = vec_add;
  auto vec_acc_pow = _mm512_mul_ps(vec_add, vec_add);
  _mm512_storeu_ps(out, vec_add);

  int i = 16;
  for (; i <= size - 16; i += 16) {
    vec_a = _loadu(a_ptr + i);
    vec_b = _loadu(b_ptr + i);
    vec_add = _mm512_add_ps(vec_a, vec_b);
    vec_acc_mean = _mm512_add_ps(vec_add, vec_acc_mean);
    _mm512_storeu_ps(out + i, vec_add);
    vec_acc_pow = _mm512_fmadd_ps(vec_add, vec_add, vec_acc_pow);
  }

  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    vec_a = _maskz_loadu(a_ptr + i, mask);
    vec_b = _maskz_loadu(b_ptr + i, mask);
    vec_add = _mm512_add_ps(vec_a, vec_b);
    auto vec_zero = _mm512_set1_ps(0);

    vec_acc_mean = _mm512_add_ps(vec_add, vec_acc_mean);
    _mm512_mask_storeu_ps(out + i, mask, vec_add);
    vec_acc_pow = _mm512_fmadd_ps(vec_add, vec_add, vec_acc_pow);
  }
  float mean_var = _mm512_reduce_add_ps(vec_acc_mean) / float(size);
  float var_val = _mm512_reduce_add_ps(vec_acc_pow);
  return std::make_pair(mean_var, var_val);
}

template <typename T, typename T1>
void _normalize_kernel(
    T* out_ptr,
    const float* input_ptr,
    const int& size,
    float scale,
    float bias,
    const T1* gamma_ptr,
    const T1* beta_ptr) {
  auto vec_one = _mm512_set1_ps(1.0);
  auto vec_zero = _mm512_set1_ps(0.0);
  auto vec_scale = _mm512_set1_ps(scale);
  auto vec_bias = _mm512_set1_ps(bias);
  int i = 0;
  for (; i <= size - 16; i += 16) {
    auto vec_input = _loadu(input_ptr + i);
    auto vec_gamma = vec_one;
    auto vec_beta = vec_zero;
    if (gamma_ptr) {
      vec_gamma = _loadu(gamma_ptr + i);
    }
    if (beta_ptr) {
      vec_beta = _loadu(beta_ptr + i);
    }
    //(a_ptr[i] * scale + bias) * gamma + beta;
    auto vec_norm = _mm512_fmadd_ps(vec_input, vec_scale, vec_bias);
    auto vec_res = _mm512_fmadd_ps(vec_norm, vec_gamma, vec_beta);
    _storeu(out_ptr + i, vec_res);
  }
  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    auto vec_input = _maskz_loadu(input_ptr + i, mask);
    auto vec_gamma = vec_one;
    auto vec_beta = vec_zero;
    if (gamma_ptr) {
      vec_gamma = _maskz_loadu(gamma_ptr + i, mask);
    }
    if (beta_ptr) {
      vec_beta = _maskz_loadu(beta_ptr + i, mask);
    }
    //(a_ptr[i] * scale + bias) * gamma + beta;
    auto vec_norm = _mm512_fmadd_ps(vec_input, vec_scale, vec_bias);
    auto vec_res = _mm512_fmadd_ps(vec_norm, vec_gamma, vec_beta);
    _mask_storeu(out_ptr + i, vec_res, mask);
  }
}

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
