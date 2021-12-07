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
namespace vec {
namespace vec512 {

using Tensor = at::Tensor;

template <typename T>
std::pair<float, float> _add_and_compute_mean_var(
    const T* a_ptr,
    const T* b_ptr,
    const int& size,
    float* out) {
  // compute add and mean/var of the value after add
  // we should firstly store add value
  auto vec_a = _load_f32_data(a_ptr);
  auto vec_b = _load_f32_data(b_ptr);
  auto vec_add = _mm512_add_ps(vec_a, vec_b);
  auto vec_acc_mean = vec_add;
  auto vec_acc_pow = _mm512_mul_ps(vec_add, vec_add);
  _mm512_store_ps(out, vec_add);

  int i = 16;
  for (; i <= size - 16; i += 16) {
    vec_a = _load_f32_data(a_ptr + i);
    vec_b = _load_f32_data(b_ptr + i);
    vec_add = _mm512_add_ps(vec_a, vec_b);
    vec_acc_mean = _mm512_add_ps(vec_add, vec_acc_mean);
    _mm512_store_ps(out + i, vec_add);
    vec_acc_pow = _mm512_fmadd_ps(vec_add, vec_add, vec_acc_pow);
  }

  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    vec_a = _maskz_load_f32_data(a_ptr + i, mask);
    vec_b = _maskz_load_f32_data(b_ptr + i, mask);
    vec_add = _mm512_add_ps(vec_a, vec_b);
    auto vec_zero = _mm512_set1_ps(0);
    _mm512_mask_store_ps(out + i, mask, vec_add);
    vec_acc_mean = _mm512_maskz_add_ps(mask, vec_add, vec_acc_mean);
    vec_acc_pow = _mm512_maskz_fmadd_ps(mask, vec_add, vec_add, vec_acc_pow);
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
    auto vec_input = _load_f32_data(input_ptr + i);
    auto vec_gamma = vec_one;
    auto vec_beta = vec_zero;
    if (gamma_ptr) {
      vec_gamma = _load_f32_data(gamma_ptr + i);
    }
    if (beta_ptr) {
      vec_beta = _load_f32_data(beta_ptr + i);
    }
    //(a_ptr[i] * scale + bias) * gamma + beta;
    auto vec_norm = _mm512_fmadd_ps(vec_input, vec_scale, vec_bias);
    auto vec_res = _mm512_fmadd_ps(vec_norm, vec_gamma, vec_beta);
    _store_data(out_ptr + i, vec_res);
  }
  if (i < size) {
    __mmask16 mask = (1 << (size - i)) - 1;
    auto vec_input = _maskz_load_f32_data(input_ptr + i, mask);
    auto vec_gamma = vec_one;
    auto vec_beta = vec_zero;
    if (!gamma_ptr) {
      vec_gamma = _maskz_load_f32_data(gamma_ptr + i, mask);
    }
    if (!beta_ptr) {
      vec_beta = _maskz_load_f32_data(beta_ptr + i, mask);
    }
    //(a_ptr[i] * scale + bias) * gamma + beta;
    auto vec_norm = _mm512_maskz_fmadd_ps(mask, vec_input, vec_scale, vec_bias);
    auto vec_res = _mm512_maskz_fmadd_ps(mask, vec_norm, vec_gamma, vec_beta);
    _mask_store_data(out_ptr + i, vec_res, mask);
  }
}

template <typename T, typename T1>
void AddLayerNormKernelImpl(
    const Tensor& a,
    const at::Tensor& b,
    int alpha,
    const Tensor& gamma,
    const Tensor& beta,
    int64_t M,
    int64_t N,
    T eps,
    Tensor& Y) {
  DCHECK_EQ(a.numel(), M * N);
  DCHECK(!gamma.defined() || gamma.numel() == N);
  DCHECK(!beta.defined() || beta.numel() == N);
  const T* a_data = a.data_ptr<T>();
  const T* b_data = b.data_ptr<T>();
  const T1* gamma_data = gamma.defined() ? gamma.data_ptr<T1>() : nullptr;
  const T1* beta_data = beta.defined() ? beta.data_ptr<T1>() : nullptr;
  T* Y_data = Y.data_ptr<T>();
  const float c = float(1) / static_cast<float>(N);
  const bool gamma_null = gamma_data == nullptr;
  const bool beta_null = beta_data == nullptr;
  at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      at::Tensor tmp_out = at::empty({N});
      float* tmp_out_ptr = tmp_out.data_ptr<float>();
      const T* a_ptr = a_data + i * N;
      const T* b_ptr = b_data + i * N;
      T* Y_ptr = Y_data + i * N;
      float mean_val;
      float rstd_val;
      std::tie(mean_val, rstd_val) =
          _add_and_compute_mean_var<T>(a_ptr, b_ptr, N, tmp_out_ptr);
      rstd_val = std::max(rstd_val * c - mean_val * mean_val, float(0));
      rstd_val = float(1.0) / std::sqrt(rstd_val + eps);
      float scale = rstd_val;
      float bias = -rstd_val * mean_val;
      _normalize_kernel<T, T1>(
          Y_ptr, tmp_out_ptr, N, scale, bias, gamma_data, beta_data);
    }
  });
}

} // namespace vec512
} // namespace vec
} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
