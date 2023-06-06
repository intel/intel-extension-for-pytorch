#include "optimizer.h"

#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

#include <cmath>
#include "omp.h"

namespace torch_ipex {
namespace cpu {

float norm_fro(const at::Tensor& input_tensor) {
  int input_size = input_tensor.numel();

  float* input_pointer = input_tensor.data_ptr<float>();
  float sum_square = 0.f;
  int num_threads = omp_get_max_threads();
  int local_size = (input_size + num_threads - 1) / num_threads;

  float scratchpad[num_threads] = {0.f};
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
  return std::sqrt(sum_square);
}

#ifdef __AVX512F__
const int Block_Size = 16;
const int Num_Blocks_Thread = 16;
const int Grid_Size = Block_Size * Num_Blocks_Thread;

float norm_fro_avx512(const at::Tensor& input_tensor) {
  int input_size = 1;
  at::IntArrayRef input_sizes = input_tensor.sizes();
  for (int i = 0; i < input_sizes.size(); i++) {
    input_size *= input_sizes[i];
  }

  float* input_pointer = input_tensor.data_ptr<float>();
  float sum_square = 0.f;

  const int Num_Grids = (input_size + Grid_Size - 1) / Grid_Size;

  float scratchpad[Num_Grids] = {0.f};

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
  return std::sqrt(sum_square);
}
#endif

/**
 * LARS fused update kernel.
 * Support Double, Float, BFloat16 training
 *@param param_ Parameters to be update
 *@param grad_ Grad used to update Parameters
 *@param momentum_buf_ momentum to accelerate convergence
 *@param param2_ Used for BF16 training, if param_ is float, param2_ is bf16
 *params need to be synced after update if param_ is BFloat16, param2_ is
 *params_ last 16 bit matissa to construct float params
 *@param momentum Args for momentum.
 *@param learning_rate  Weight for grad while update.
 *@param eeta Trust coefficient
 *@param eps Prevent division by zero
 *@param weight_decay Args for regularization to avoid over-fit.
 *@param dampening Attribute for momentum.
 *@param nesterov Attribute for momentum.
 */
c10::optional<at::Tensor> lars_fused_step(
    at::Tensor& param_,
    const at::Tensor& grad_,
    const c10::optional<at::Tensor>& momentum_buf_,
    at::Tensor& param2_,
    double momentum,
    double learning_rate,
    double eeta,
    double eps,
    double weight_decay,
    double dampening,
    bool nesterov) {
  RECORD_FUNCTION(
      "torch_ipex::lars_fused_step", c10::ArrayRef<c10::IValue>({}));

  TORCH_CHECK(
      weight_decay >= 0, "Expect weight_decay >= 0.0, got ", weight_decay);

  TORCH_CHECK(
      param_.sizes() == grad_.sizes(),
      "Expect param and grad_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; grad_ sizes: ",
      grad_.sizes());
  TORCH_CHECK(
      !momentum_buf_.has_value() ||
          param_.sizes() == momentum_buf_.value().sizes(),
      "Expect param and momentum_buf have the same sizes, param sizes: ",
      param_.sizes(),
      "; momentum_buf sizes: ",
      momentum_buf_.value().sizes());
  TORCH_CHECK(
      param2_.numel() == 0 || param_.sizes() == param2_.sizes(),
      "Expect param and param2_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; param2_ sizes: ",
      param2_.sizes());

  at::Tensor grad_f32 = grad_.to(torch::kFloat32);
#ifdef __AVX512F__
  float w_norm = norm_fro_avx512(param_);
  float g_norm = norm_fro_avx512(grad_f32);
#else
  float w_norm = norm_fro(param_);
  float g_norm = norm_fro(grad_f32);
#endif

  float trust_ratio = 1.f;
  if ((w_norm > 0) && (g_norm > 0)) {
    trust_ratio = eeta * w_norm / (g_norm + weight_decay * w_norm + eps);
  }
  learning_rate *= trust_ratio;
  return sgd_fused_step_kernel_stub(
      kCPU,
      param_,
      grad_,
      momentum_buf_,
      param2_,
      momentum,
      learning_rate,
      weight_decay,
      dampening,
      nesterov);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "lars_fused_step(Tensor param, Tensor grad, Tensor? momentum_buf, Tensor "
      "trail, float momentum, float learning_rate, float eeta, float eps,"
      "float weight_decay, float dampening, bool nesterov) -> Tensor?",
      torch_ipex::cpu::lars_fused_step);
}

} // namespace
