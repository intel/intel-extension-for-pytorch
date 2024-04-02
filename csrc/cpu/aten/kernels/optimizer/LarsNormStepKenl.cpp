#include <ATen/Parallel.h>
#include <aten/optimizer/optimizer.h>
#include <omp.h>
#include <torch/csrc/autograd/function.h>
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

float lars_norm_kernel_impl(const at::Tensor& input_tensor_) {
  auto input_tensor = input_tensor_.contiguous();
  float* input_pointer = input_tensor.data_ptr<float>();
  float sum_square = 0.f;
  auto input_size = input_tensor.numel();

#if defined(CPU_CAPABILITY_AVX512)
  const int Block_Size = 16;
  const int Num_Blocks_Thread = 16;
  const int Grid_Size = Block_Size * Num_Blocks_Thread;
  const int Num_Grids = (input_size + Grid_Size - 1) / Grid_Size;
  std::vector<float> scratchpad(Num_Grids, 0.f);

#pragma omp parallel for
  for (int grid = 0; grid < Num_Grids; grid++) {
    int local_start = grid * Grid_Size;
    float* local_pointer = input_pointer + local_start;
    int local_ind = 0;
    __m512 acc_reg = _mm512_setzero_ps();
    __m512 mul_reg;
    while ((local_ind + Block_Size - 1 < Grid_Size) &&
           (local_start + local_ind + Block_Size - 1 < input_size)) {
      mul_reg = _mm512_load_ps(local_pointer + local_ind);
      acc_reg = _mm512_fmadd_ps(mul_reg, mul_reg, acc_reg);
      local_ind += Block_Size;
    }
    float local_value = _mm512_reduce_add_ps(acc_reg);
    while ((local_ind < Grid_Size) && (local_start + local_ind < input_size)) {
      local_value += local_pointer[local_ind] * local_pointer[local_ind];
      local_ind++;
    }
    scratchpad[grid] = local_value;
  }
  for (int i = 0; i < Num_Grids; i++) {
    sum_square += scratchpad[i];
  }
#else
  int num_threads = omp_get_max_threads();
  int local_size = (input_size + num_threads - 1) / num_threads;

  std::vector<float> scratchpad(num_threads, 0.f);

// Reduce to scratchpad
#pragma omp parallel
  {
    int threadId = omp_get_thread_num();
    int local_start = local_size * threadId;
    float* local_pointer = input_pointer + local_start;
    float local_value = 0.f;
    int local_ind = 0;
    while ((local_ind < local_size) && (local_start + local_ind < input_size)) {
      local_value += local_pointer[local_ind] * local_pointer[local_ind];

      local_ind++;
    }
    scratchpad[threadId] = local_value;
  }
  for (int i = 0; i < num_threads; i++) {
    sum_square += scratchpad[i];
  }
#endif
  return std::sqrt(sum_square);
}

} // anonymous namespace

IPEX_REGISTER_DISPATCH(lars_norm_kernel_stub, &lars_norm_kernel_impl);

} // namespace cpu
} // namespace torch_ipex