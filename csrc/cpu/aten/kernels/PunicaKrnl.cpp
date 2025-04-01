#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/Tensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/utils.h>
#include <aten/Punica.h>
#include <aten/utils/mkl_gemm.h>
#include <c10/util/irange.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <omp.h>
#include <limits>
#include "csrc/cpu/tpp/woq/tla.h"
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

template <typename T1, typename T2>
void _dot(
    const T1* intput,
    const T2* weight,
    T1* out,
    int64_t len,
    const double scale,
    bool add_inputs) {
  using namespace torch_ipex::cpu::kernel;
  int64_t hsi = 0;
  float out_f32 = 0;
#if defined(CPU_CAPABILITY_AVX512)
  int64_t vec_size = 16; // 512/32
  auto qk_sum_vec = _mm512_setzero_ps();
  for (hsi = 0; hsi <= len - vec_size; hsi += vec_size) {
    auto q_vec = _loadu(intput + hsi);
    auto k_vec = _loadu(weight + hsi);
    qk_sum_vec = _mm512_fmadd_ps(q_vec, k_vec, qk_sum_vec);
  }
  out_f32 += _mm512_reduce_add_ps(qk_sum_vec);
#endif
  for (; hsi < len; hsi++) {
    out_f32 +=
        static_cast<float>(intput[hsi]) * static_cast<float>(weight[hsi]);
  }
  if (add_inputs) {
    out[0] += static_cast<T1>(out_f32 * scale);
  } else {
    out[0] = static_cast<T1>(out_f32 * scale);
  }
}

template <typename T>
void punica_bgmv_expand_slice_kernel(
    at::Tensor&
        out, // [bs, output_size1] output_size1 >= slice_offset + slice_size
    at::Tensor& input, // [bs, max_rank]
    at::Tensor& weights, // [num_lora, hidden_size, max_rank]
    at::Tensor& indicies, // [bs]
    int64_t slice_offset,
    int64_t slice_size,
    bool add_inputs) {
  int64_t num_lora = weights.size(0);
  int64_t hidden_size = weights.size(1);
  int64_t max_rank = weights.size(2);
  int64_t batch_size = out.size(0);
  int64_t output_size1 = out.size(1);
  int64_t input_size1 = input.size(1);
  TORCH_CHECK(input_size1 == max_rank);
  TORCH_CHECK(slice_offset >= 0)
  TORCH_CHECK(slice_size == hidden_size)
  TORCH_CHECK(output_size1 >= slice_offset + slice_size);
  TORCH_CHECK(batch_size == indicies.size(0));
  TORCH_CHECK(batch_size == input.size(0));
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(weights.is_contiguous());
  TORCH_CHECK(indicies.is_contiguous());
  TORCH_CHECK(out.is_contiguous());

  int64_t* indicies_ptr = indicies.data_ptr<int64_t>();
  T* out_ptr = out.data_ptr<T>();
  T* input_ptr = input.data_ptr<T>();
  T* weights_ptr = weights.data_ptr<T>();
  bool limit = (input.size(0) == 1 && batch_size != 0);

#pragma omp parallel for collapse(2) schedule(static, 1)
  for (int64_t bs = 0; bs < batch_size; bs++) {
    for (int64_t h = 0; h < hidden_size; h++) {
      int64_t input_bs = limit ? 0 : bs;
      int64_t weights_offset =
          indicies_ptr[bs] * max_rank * hidden_size + h * max_rank;
      T* weight_start = weights_ptr + weights_offset;
      T* input_start = input_ptr + input_bs * input_size1;
      T* out_start = out_ptr + bs * output_size1 + h + slice_offset;
      _dot<T, T>(input_start, weight_start, out_start, max_rank, 1, add_inputs);
    }
  }
}

template <typename T>
void punica_bgmv_shrink_kernel(
    at::Tensor& out, // [bs, output_size1] output_size1 >= max_rank
    at::Tensor& input, // [bs, input_size1]  input_size1  >= hidden_size
    at::Tensor& weights, // [num_lora, max_rank, hidden_size]
    at::Tensor& indicies, // [bs]
    const double scale) {
  int64_t num_lora = weights.size(0);
  int64_t max_rank = weights.size(1);
  int64_t hidden_size = weights.size(2);
  int64_t batch_size = out.size(0);
  int64_t output_size1 = out.size(1);
  int64_t input_size1 = input.size(1);
  TORCH_CHECK(input_size1 >= hidden_size);
  TORCH_CHECK(output_size1 >= max_rank);
  TORCH_CHECK(batch_size == input.size(0));
  TORCH_CHECK(batch_size == indicies.size(0));
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(weights.is_contiguous());
  TORCH_CHECK(indicies.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  int64_t* indicies_ptr = indicies.data_ptr<int64_t>();
  T* out_ptr = out.data_ptr<T>();
  T* input_ptr = input.data_ptr<T>();
  T* weights_ptr = weights.data_ptr<T>();

#pragma omp parallel for collapse(2) schedule(static, 1)
  for (int64_t bs = 0; bs < batch_size; bs++) {
    for (int64_t r = 0; r < max_rank; r++) {
      int64_t weights_offset =
          indicies_ptr[bs] * max_rank * hidden_size + r * hidden_size;
      T* weight_start = weights_ptr + weights_offset;
      T* input_start = input_ptr + bs * input_size1;
      T* out_start = out_ptr + bs * output_size1 + r;
      _dot<T, T>(
          input_start, weight_start, out_start, hidden_size, scale, false);
    }
  }
}

template <typename T>
void punica_bgmv_expand_kernel(
    at::Tensor& out, // [bs, output_size1] output_size1 >= max_rank
    at::Tensor& input, // [bs, input_size1]  input_size1  >= hidden_size
    at::Tensor& weights, // [num_lora, max_rank, hidden_size]
    at::Tensor& indicies, // [bs]
    bool add_inputs) {
  int64_t num_lora = weights.size(0);
  int64_t max_rank = weights.size(1);
  int64_t hidden_size = weights.size(2);
  int64_t batch_size = out.size(0);
  int64_t output_size1 = out.size(1);
  int64_t input_size1 = input.size(1);
  TORCH_CHECK(input_size1 >= hidden_size);
  TORCH_CHECK(output_size1 >= max_rank);
  TORCH_CHECK(batch_size == indicies.size(0));
  TORCH_CHECK(batch_size == input.size(0) || input.size(0) == 1);
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(weights.is_contiguous());
  TORCH_CHECK(indicies.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  int64_t* indicies_ptr = indicies.data_ptr<int64_t>();
  T* out_ptr = out.data_ptr<T>();
  T* input_ptr = input.data_ptr<T>();
  T* weights_ptr = weights.data_ptr<T>();
  bool limit = (input.size(0) == 1 && batch_size != 0);

#pragma omp parallel for collapse(2) schedule(static, 1)
  for (int64_t bs = 0; bs < batch_size; bs++) {
    for (int64_t r = 0; r < max_rank; r++) {
      int64_t input_bs = limit ? 0 : bs;
      int64_t weights_offset =
          indicies_ptr[bs] * max_rank * hidden_size + r * hidden_size;
      T* weight_start = weights_ptr + weights_offset;
      T* input_start = input_ptr + input_bs * input_size1;
      T* out_start = out_ptr + bs * output_size1 + r;
      _dot<T, T>(
          input_start, weight_start, out_start, hidden_size, 1, add_inputs);
    }
  }
}

void punica_bgmv_shrink_kernel_impl(
    at::Tensor& out, // [bs, output_size1] output_size1 >= max_rank
    at::Tensor& input, // [bs, input_size1]  input_size1  >= hidden_size
    at::Tensor& weights, // [num_lora, max_rank, hidden_size]
    at::Tensor& indicies, // [bs]
    const double scale) {
  RECORD_FUNCTION(
      "ipex::punica_bgmv_shrink_kernel_impl", c10::ArrayRef<c10::IValue>({}));
  TORCH_CHECK(
      weights.scalar_type() == out.scalar_type(),
      "dtype of weight and out must be same");
  TORCH_CHECK(
      input.scalar_type() == out.scalar_type(),
      "dtype of input and out must be same");
  TORCH_CHECK(out.dim() == 2, "out must be 2D");
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(weights.dim() == 3, "weights must be 3D");
  TORCH_CHECK(indicies.dim() == 1, "indicies must be 1D");
  if (out.scalar_type() == at::kBFloat16) {
    punica_bgmv_shrink_kernel<at::BFloat16>(
        out, input, weights, indicies, scale);
  } else if (out.scalar_type() == at::kHalf) {
    punica_bgmv_shrink_kernel<at::Half>(out, input, weights, indicies, scale);
  }
}

void punica_bgmv_expand_kernel_impl(
    at::Tensor& out, // [bs, output_size1] output_size1 >= max_rank
    at::Tensor& input, // [bs, input_size1] or [1, input_size1] input_size1  >=
                       // hidden_size
    at::Tensor& weights, // [num_lora, max_rank, hidden_size]
    at::Tensor& indicies, // [bs]
    bool add_inputs) {
  RECORD_FUNCTION(
      "ipex::punica_bgmv_expand_kernel_impl", c10::ArrayRef<c10::IValue>({}));
  TORCH_CHECK(
      weights.scalar_type() == out.scalar_type(),
      "dtype of weight and out must be same");
  TORCH_CHECK(
      input.scalar_type() == out.scalar_type(),
      "dtype of input and out must be same");
  TORCH_CHECK(out.dim() == 2, "out must be 2D");
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(weights.dim() == 3, "weights must be 3D");
  TORCH_CHECK(indicies.dim() == 1, "indicies must be 1D");
  if (out.scalar_type() == at::kBFloat16) {
    punica_bgmv_expand_kernel<at::BFloat16>(
        out, input, weights, indicies, add_inputs);
  } else if (out.scalar_type() == at::kHalf) {
    punica_bgmv_expand_kernel<at::Half>(
        out, input, weights, indicies, add_inputs);
  }
}

void punica_bgmv_expand_slice_kernel_impl(
    at::Tensor& out, // [bs, output_size1] output_size1 >= max_rank
    at::Tensor& input, // [bs, input_size1]  input_size1  >= hidden_size
    at::Tensor& weights, // [num_lora, max_rank, hidden_size]
    at::Tensor& indicies, // [bs]
    int64_t slice_offset,
    int64_t slice_size,
    bool add_inputs) {
  RECORD_FUNCTION(
      "ipex::punica_bgmv_expand_slice_kernel_impl",
      c10::ArrayRef<c10::IValue>({}));
  TORCH_CHECK(
      weights.scalar_type() == out.scalar_type(),
      "dtype of weight and out must be same");
  TORCH_CHECK(
      input.scalar_type() == out.scalar_type(),
      "dtype of input and out must be same");
  TORCH_CHECK(out.dim() == 2, "out must be 2D");
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(weights.dim() == 3, "weights must be 3D");
  TORCH_CHECK(indicies.dim() == 1, "indicies must be 1D");
  if (out.scalar_type() == at::kBFloat16) {
    punica_bgmv_expand_slice_kernel<at::BFloat16>(
        out, input, weights, indicies, slice_offset, slice_size, add_inputs);
  } else if (out.scalar_type() == at::kHalf) {
    punica_bgmv_expand_slice_kernel<at::Half>(
        out, input, weights, indicies, slice_offset, slice_size, add_inputs);
  }
}

} // namespace

IPEX_REGISTER_DISPATCH(
    punica_bgmv_shrink_kernel_stub,
    &punica_bgmv_shrink_kernel_impl);

IPEX_REGISTER_DISPATCH(
    punica_bgmv_expand_kernel_stub,
    &punica_bgmv_expand_kernel_impl);

IPEX_REGISTER_DISPATCH(
    punica_bgmv_expand_slice_kernel_stub,
    &punica_bgmv_expand_slice_kernel_impl);
} // namespace cpu
} // namespace torch_ipex