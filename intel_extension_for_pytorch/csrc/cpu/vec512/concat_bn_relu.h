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

namespace torch_ipex {
namespace cpu {
namespace kernel {
namespace vec {
namespace vec512 {

using Tensor = at::Tensor;

template <typename T, typename ACC_T>
static void _concat_bn_relu_kernel_channels_last(
    const std::vector<const T*>& in_ptr,
    const std::vector<int64_t>& in_ch,
    T* out_ptr,
    const ACC_T* scale_ptr,
    const ACC_T* beta_ptr,
    int64_t total_size_except_channels,
    int64_t ci,
    int64_t co) {
  int64_t i = 0, j = 0, k = 0;
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
  for (i = 0; i < total_size_except_channels; ++i) {
    for (j = 0; j < in_ptr.size(); ++j) {
      auto concat_in_ptr = in_ptr[j] + i * in_ch[j + 1] - (i + 1) * in_ch[j];
      for (k = in_ch[j]; k < in_ch[j + 1]; k += 16) {
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

template <>
void _concat_bn_relu_kernel_channels_last<at::BFloat16, float>(
    const std::vector<const at::BFloat16*>& in_ptr,
    const std::vector<int64_t>& in_ch,
    at::BFloat16* out_ptr,
    const float* scale_ptr,
    const float* beta_ptr,
    int64_t total_size_except_channels,
    int64_t ci,
    int64_t co) {
  int64_t i = 0, j = 0, k = 0;
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
  for (i = 0; i < total_size_except_channels; ++i) {
    for (j = 0; j < in_ptr.size(); ++j) {
      auto concat_in_ptr = in_ptr[j] + i * in_ch[j + 1] - (i + 1) * in_ch[j];
      for (k = in_ch[j]; k < in_ch[j + 1]; k += 16) {
        auto in =
            cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(concat_in_ptr + k)));
        auto beta = _mm512_loadu_ps(beta_ptr + k);
        auto scale = _mm512_loadu_ps(scale_ptr + k);
        auto bn_out = _mm512_add_ps(beta, _mm512_mul_ps(scale, in));
        auto out = _mm512_max_ps(zero, bn_out);
        _mm256_storeu_si256(
            (__m256i*)(out_ptr + i * co + k), cvt_fp32_to_bf16(out));
      }
    }
  }
}

//  All the fusion conditions have been applied before calling this kernel.
//  Please refer ../../jit/cpu/passes/graph_rewrite.cpp for details.
template <typename T>
void ConcatBnReluKernelImpl_ChannelsLast(
    const c10::List<Tensor>& a,
    const Tensor& scale,
    const Tensor& beta,
    Tensor& output) {
  using ACC_T = typename AccType<T>::type;
  int64_t list_length = a.size();
  int64_t total_size_except_channels = 1;
  std::vector<const T*> input_ptr(list_length);
  std::vector<int64_t> input_channels(list_length + 1);

  for (int64_t i = 0; i < list_length; ++i) {
    input_channels[i + 1] = input_channels[i] + a[i].size(1);
    input_ptr[i] = a[i].contiguous(a[i].suggest_memory_format()).data_ptr<T>();
  }
  //  Return the product of all the input dimensions except for the channel
  //  and check if the dimension and sizes of the tensors meet the fusion
  //  requirements.
  for (int64_t i = 0; i < a[0].ndimension(); ++i) {
    if (i != 1)
      total_size_except_channels *= a[0].size(i);
  }

  const ACC_T* scale_data = scale.data_ptr<ACC_T>();
  const ACC_T* beta_data = beta.data_ptr<ACC_T>();
  T* output_data = output.data_ptr<T>();

  _concat_bn_relu_kernel_channels_last<T, ACC_T>(
      input_ptr,
      input_channels,
      output_data,
      scale_data,
      beta_data,
      total_size_except_channels,
      a[0].size(1),
      output.size(1));
}

} // namespace vec512
} // namespace vec
} // namespace kernel
} // namespace cpu
} // namespace torch_ipex