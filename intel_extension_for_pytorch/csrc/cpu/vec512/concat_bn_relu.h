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
    std::vector<const T*>& in_ptr,
    T* out_ptr,
    const T* scale_ptr,
    const T* beta_ptr,
    int64_t total_size,
    int64_t ci,
    int64_t co) {
  int64_t i = 0, j = 0, k = 0;
  auto zero = _mm512_set1_ps(0.0);
#pragma omp parallel for collapse(3)
  for (i = 0; i < total_size; ++i) {
    for (j = 0; j < ci; j += 16) {
      for (k = 0; k < in_ptr.size(); ++k) {
        _mm512_store_ps(
            out_ptr + i * co + j + ci * k,
            _mm512_max_ps(
                zero,
                _mm512_add_ps(
                    _mm512_load_ps(beta_ptr + j + ci * k),
                    _mm512_mul_ps(
                        _mm512_load_ps(scale_ptr + j + ci * k),
                        _mm512_load_ps(in_ptr[k] + i * ci + j)))));
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
  int64_t input_len = a.size();
  int64_t total_size = 1;
  std::vector<const T*> input_ptr;

  //  Return the product of all the input dimensions except for the channel
  //  and check if the dimension and sizes of the tensors meet the fusion
  //  requirements.
  for (int64_t i = 0; i < a[0].ndimension(); ++i) {
    if (i != 1)
      total_size *= a[0].size(i);
  }

  //  The condition of calling this kernel includes that the memory format
  //  of all the input & output tensors should be ChannelsLast.
  //  Thus here the contiguous is applied to ensure the continuity.
  auto memory_format = a[0].ndimension() == 4
      ? (at::MemoryFormat::ChannelsLast)
      : (at::MemoryFormat::ChannelsLast3d);
  for (int64_t i = 0; i < input_len; ++i) {
    input_ptr.push_back(a[i].contiguous(memory_format).data_ptr<T>());
  }
  const T* scale_data = scale.data_ptr<T>();
  const T* beta_data = beta.data_ptr<T>();
  T* output_data = output.contiguous(memory_format).data_ptr<T>();

  _concat_bn_relu_kernel_channels_last<T>(
      input_ptr,
      output_data,
      scale_data,
      beta_data,
      total_size,
      a[0].size(1),
      output.size(1));
}

} // namespace vec512
} // namespace vec
} // namespace kernel
} // namespace cpu
} // namespace torch_ipex
