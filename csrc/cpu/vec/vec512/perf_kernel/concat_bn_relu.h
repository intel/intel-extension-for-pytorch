#pragma once

#include <immintrin.h>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <c10/util/SmallVector.h>
#include <limits>
#include "utils.h"

// use float as accumulation type for BFloat16
template <typename scalar_t>
struct AccType {
  using type = scalar_t;
};
template <>
struct AccType<BFloat16> {
  using type = float;
};
template <>
struct AccType<Half> {
  using type = float;
};

namespace torch_ipex {
namespace cpu {
namespace kernel {

using Tensor = at::Tensor;

template <typename T, typename ACC_T>
static typename std::enable_if<!is_reduced_floating_point_v<T>, void>::type
_concat_bn_relu_kernel_channels_last(
    const std::vector<const T*>& in_ptr,
    const std::vector<int64_t>& in_ch,
    T* out_ptr,
    const ACC_T* scale_ptr,
    const ACC_T* beta_ptr,
    int64_t total_size_except_channels,
    int64_t ci,
    int64_t co) {
  auto zero = _mm512_set1_ps(0.0);
#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
#pragma omp parallel for schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for (int64_t i = 0; i < total_size_except_channels; ++i) {
    for (int64_t j = 0; j < in_ptr.size(); ++j) {
      auto concat_in_ptr = in_ptr[j] + i * in_ch[j + 1] - (i + 1) * in_ch[j];
      for (int64_t k = in_ch[j]; k < in_ch[j + 1]; k += 16) {
        auto in = _mm512_loadu_ps(concat_in_ptr + k);
        auto beta = _mm512_loadu_ps(beta_ptr + k);
        auto scale = _mm512_loadu_ps(scale_ptr + k);
        auto bn_out = _mm512_add_ps(beta, _mm512_mul_ps(scale, in));
        auto out = _mm512_max_ps(zero, bn_out);
        _mm512_storeu_ps(out_ptr + i * co + k, out);
      }
    }
  }
}

template <typename T, typename ACC_T>
static typename std::enable_if<is_reduced_floating_point_v<T>, void>::type
_concat_bn_relu_kernel_channels_last(
    const std::vector<const T*>& in_ptr,
    const std::vector<int64_t>& in_ch,
    T* out_ptr,
    const float* scale_ptr,
    const float* beta_ptr,
    int64_t total_size_except_channels,
    int64_t ci,
    int64_t co) {
  auto zero = _mm512_set1_ps(0.0);
#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#else
#pragma omp parallel for schedule( \
    static) if (omp_get_max_threads() > 1 && !omp_in_parallel())
#endif
#endif
  for (int64_t i = 0; i < total_size_except_channels; ++i) {
    for (int64_t j = 0; j < in_ptr.size(); ++j) {
      auto concat_in_ptr = in_ptr[j] + i * in_ch[j + 1] - (i + 1) * in_ch[j];
      for (int64_t k = in_ch[j]; k < in_ch[j + 1]; k += 16) {
        auto in =
            cvt_to_fp32<T>(_mm256_loadu_si256((__m256i*)(concat_in_ptr + k)));
        auto beta = _mm512_loadu_ps(beta_ptr + k);
        auto scale = _mm512_loadu_ps(scale_ptr + k);
        auto bn_out = _mm512_add_ps(beta, _mm512_mul_ps(scale, in));
        auto out = _mm512_max_ps(zero, bn_out);
        _mm256_storeu_si256(
            (__m256i*)(out_ptr + i * co + k), cvt_from_fp32<T>(out));
      }
    }
  }
}

} // namespace kernel
} // namespace cpu
} // namespace torch_ipex