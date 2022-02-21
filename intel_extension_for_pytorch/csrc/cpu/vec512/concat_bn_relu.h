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
void _concat_bn_relu_kernel_channels_last(
    const std::vector<const T*>& in_ptr,
    const std::vector<int64_t>& in_ch,
    T* out_ptr,
    const T* scale_ptr,
    const T* beta_ptr,
    int64_t total_size_except_channels,
    int64_t ci,
    int64_t co) {
  int64_t i = 0, j = 0, k = 0;
  auto zero = _mm512_set1_ps(0.0);
#pragma omp parallel for collapse(2)
  for (i = 0; i < total_size_except_channels; ++i) {
    for (j = 0; j < in_ptr.size(); ++j) {
      for (k = in_ch[j]; k < in_ch[j + 1]; k += 16) {
        _mm512_store_ps(
            out_ptr + i * co + k,
            _mm512_max_ps(
                zero,
                _mm512_add_ps(
                    _mm512_load_ps(beta_ptr + k),
                    _mm512_mul_ps(
                        _mm512_load_ps(scale_ptr + k),
                        _mm512_load_ps(
                            in_ptr[j] + i * (in_ch[j + 1] - in_ch[j]) + k -
                            in_ch[j])))));
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

  const T* scale_data = scale.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  T* output_data = output.data_ptr<T>();

  _concat_bn_relu_kernel_channels_last<T>(
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