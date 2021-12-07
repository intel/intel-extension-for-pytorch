#include "csrc/cpu/vec512/bf16/vec/vec_type_cvt.h"
#include "optimizer.h"

#include <torch/csrc/autograd/function.h>
#include <torch/extension.h>
namespace torch_ipex {
namespace cpu {

using namespace at::vec;

template <typename scalar_t, typename grad_t>
void adagrad_fused_step_kernel(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& state_sum,
    const at::Tensor& param2,
    int64_t step,
    double learning_rate,
    double weight_decay,
    double lr_decay,
    double eps) {
  scalar_t* param_data = param.data_ptr<scalar_t>();
  scalar_t* grad_data = grad.data_ptr<scalar_t>();
  scalar_t* state_sum_data = state_sum.data_ptr<scalar_t>();

  // update learning rate
  double clr = learning_rate / (1 + (step - 1) * lr_decay);

  using Vec = at::vec::Vectorized<scalar_t>;

  int64_t grain_size = 512;

  // purely element-wise operations
  at::parallel_for(
      0, param.numel(), grain_size, [&](int64_t begin, int64_t end) {
        // local pointers
        scalar_t* param_ptr = param_data + begin;
        scalar_t* grad_ptr = grad_data + begin;
        scalar_t* state_sum_ptr = state_sum_data + begin;

        const int64_t size = end - begin;

        int64_t d = 0;
        for (; d < size - (size % Vec::size()); d += Vec::size()) {
          Vec param_vec = Vec::loadu(param_ptr + d);
          Vec grad_vec = Vec::loadu(grad_ptr + d) +
              param_vec * Vec(scalar_t(weight_decay));

          Vec sum_vec = Vec::loadu(state_sum_ptr + d) + grad_vec * grad_vec;
          sum_vec.store(state_sum_ptr + d);

          Vec std_vec = sum_vec.sqrt() + Vec(scalar_t(eps));
          param_vec = param_vec - grad_vec / std_vec * Vec(scalar_t(clr));
          param_vec.store(param_ptr + d);
        }
        for (; d < size; d++) {
          scalar_t grad_val = grad_ptr[d] + param_ptr[d] * weight_decay;
          state_sum_ptr[d] += grad_val * grad_val;

          scalar_t std_val = std::sqrt(state_sum_ptr[d]) + eps;
          param_ptr[d] -= grad_val / std_val * clr;
        }
      });
}

template <>
void adagrad_fused_step_kernel<at::BFloat16, at::BFloat16>(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& state_sum,
    const at::Tensor& param2,
    int64_t step,
    double learning_rate,
    double weight_decay,
    double lr_decay,
    double eps) {
  TORCH_CHECK(
      param.scalar_type() == at::kBFloat16,
      "adagrad_fused_step_kernel: expect param to be at::BFloat16");
  TORCH_CHECK(
      grad.scalar_type() == at::kBFloat16,
      "adagrad_fused_step_kernel: expect grad to be at::BFloat16");
  TORCH_CHECK(
      state_sum.scalar_type() == at::kFloat,
      "adagrad_fused_step_kernel: expect stats_sum to be float32");
  TORCH_CHECK(
      param2.scalar_type() == at::kBFloat16,
      "adagrad_fused_step_kernel: expect param2 to be at::BFloat16");

  at::BFloat16* param_data = param.data_ptr<at::BFloat16>();
  at::BFloat16* grad_data = grad.data_ptr<at::BFloat16>();
  float* state_sum_data = state_sum.data_ptr<float>();
  at::BFloat16* param2_data = param2.data_ptr<at::BFloat16>();

  // update learning rate
  double clr = learning_rate / (1 + (step - 1) * lr_decay);

  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;

  int64_t grain_size = 512;

  // purely element-wise operations
  at::parallel_for(
      0, param.numel(), grain_size, [&](int64_t begin, int64_t end) {
        // local pointers
        at::BFloat16* param_ptr = param_data + begin;
        at::BFloat16* grad_ptr = grad_data + begin;
        float* state_sum_ptr = state_sum_data + begin;
        at::BFloat16* param2_ptr = param2_data + begin;

        const int64_t size = end - begin;

        int64_t d = 0;
        for (; d < size - (size % bVec::size()); d += bVec::size()) {
          bVec param_bvec = bVec::loadu(param_ptr + d);
          bVec param2_bvec = bVec::loadu(param2_ptr + d);
          fVec param_fvec, param_fvec2;
          std::tie(param_fvec, param_fvec2) =
              bf16::pack_bfloat16_float(param_bvec, param2_bvec);

          bVec grad_bvec = bVec::loadu(grad_ptr + d);
          fVec grad_fvec, grad_fvec2;
          std::tie(grad_fvec, grad_fvec2) = convert_bfloat16_float(grad_bvec);

          grad_fvec = grad_fvec + param_fvec * fVec(float(weight_decay));
          grad_fvec2 = grad_fvec2 + param_fvec2 * fVec(float(weight_decay));

          fVec sum_fvec =
              fVec::loadu(state_sum_ptr + d) + grad_fvec * grad_fvec;
          fVec sum_fvec2 = fVec::loadu(state_sum_ptr + d + fVec::size()) +
              grad_fvec2 * grad_fvec2;
          sum_fvec.store(state_sum_ptr + d);
          sum_fvec2.store(state_sum_ptr + d + fVec::size());

          fVec std_fvec = sum_fvec.sqrt() + fVec(float(eps));
          fVec std_fvec2 = sum_fvec2.sqrt() + fVec(float(eps));
          param_fvec = param_fvec - grad_fvec / std_fvec * fVec(float(clr));
          param_fvec2 = param_fvec2 - grad_fvec2 / std_fvec2 * fVec(float(clr));

          std::tie(param_bvec, param2_bvec) =
              bf16::unpack_float_bfloat16(param_fvec, param_fvec2);
          param_bvec.store(param_ptr + d);
          param2_bvec.store(param2_ptr + d);
        }
        for (; d < size; d++) {
          float param_val =
              bf16::pack_bfloat16_float(param_ptr[d], param2_ptr[d]);
          float grad_val = float(grad_ptr[d]) + param_val * weight_decay;
          state_sum_ptr[d] += grad_val * grad_val;

          float std_val = std::sqrt(state_sum_ptr[d]) + eps;
          param_val -= grad_val / std_val * clr;
          std::tie(param_ptr[d], param2_ptr[d]) =
              bf16::unpack_float_bfloat16(param_val);
        }
      });
}

template <>
void adagrad_fused_step_kernel<float, at::BFloat16>(
    const at::Tensor& param,
    const at::Tensor& grad,
    const at::Tensor& state_sum,
    const at::Tensor& param2,
    int64_t step,
    double learning_rate,
    double weight_decay,
    double lr_decay,
    double eps) {
  TORCH_CHECK(
      param.scalar_type() == at::kFloat,
      "adagrad_fused_step_kernel: expect param to be float32");
  TORCH_CHECK(
      grad.scalar_type() == at::kBFloat16,
      "adagrad_fused_step_kernel: expect grad to be at::BFloat16");
  TORCH_CHECK(
      state_sum.scalar_type() == at::kFloat,
      "adagrad_fused_step_kernel: expect stats_sum to be float32");
  TORCH_CHECK(
      param2.scalar_type() == at::kBFloat16,
      "adagrad_fused_step_kernel: expect param2 to be at::BFloat16");

  float* param_data = param.data_ptr<float>();
  at::BFloat16* grad_data = grad.data_ptr<at::BFloat16>();
  float* state_sum_data = state_sum.data_ptr<float>();
  at::BFloat16* param2_data = param2.data_ptr<at::BFloat16>();

  // update learning rate
  double clr = learning_rate / (1 + (step - 1) * lr_decay);

  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;

  int64_t grain_size = 512;

  // purely element-wise operations
  at::parallel_for(
      0, param.numel(), grain_size, [&](int64_t begin, int64_t end) {
        // local pointers
        float* param_ptr = param_data + begin;
        at::BFloat16* grad_ptr = grad_data + begin;
        float* state_sum_ptr = state_sum_data + begin;
        at::BFloat16* param2_ptr = param2_data + begin;

        const int64_t size = end - begin;

        int64_t d = 0;
        for (; d < size - (size % bVec::size()); d += bVec::size()) {
          fVec param_fvec = fVec::loadu(param_ptr + d);
          fVec param_fvec2 = fVec::loadu(param_ptr + d + fVec::size());

          bVec grad_bvec = bVec::loadu(grad_ptr + d);
          fVec grad_fvec, grad_fvec2;
          std::tie(grad_fvec, grad_fvec2) = convert_bfloat16_float(grad_bvec);

          grad_fvec = grad_fvec + param_fvec * fVec(float(weight_decay));
          grad_fvec2 = grad_fvec2 + param_fvec2 * fVec(float(weight_decay));

          fVec sum_fvec =
              fVec::loadu(state_sum_ptr + d) + grad_fvec * grad_fvec;
          fVec sum_fvec2 = fVec::loadu(state_sum_ptr + d + fVec::size()) +
              grad_fvec2 * grad_fvec2;
          sum_fvec.store(state_sum_ptr + d);
          sum_fvec2.store(state_sum_ptr + d + fVec::size());

          fVec std_fvec = sum_fvec.sqrt() + fVec(float(eps));
          fVec std_fvec2 = sum_fvec2.sqrt() + fVec(float(eps));
          param_fvec = param_fvec - grad_fvec / std_fvec * fVec(float(clr));
          param_fvec2 = param_fvec2 - grad_fvec2 / std_fvec2 * fVec(float(clr));

          param_fvec.store(param_ptr + d);
          param_fvec2.store(param_ptr + d + fVec::size());
          // sync float param to bfloat16
          bVec param2_bvec = convert_float_bfloat16(param_fvec, param_fvec2);
          param2_bvec.store(param2_ptr + d);
        }
        for (; d < size; d++) {
          float param_val =
              bf16::pack_bfloat16_float(param_ptr[d], param2_ptr[d]);
          float grad_val = float(grad_ptr[d]) + param_val * weight_decay;
          state_sum_ptr[d] += grad_val * grad_val;

          float std_val = std::sqrt(state_sum_ptr[d]) + eps;
          param_val -= grad_val / std_val * clr;
          param_ptr[d] = param_val;
          param2_ptr[d] = at::BFloat16(param_val);
        }
      });
}

std::tuple<at::Tensor, at::Tensor> adagrad_fused_step(
    const at::Tensor& param_,
    const at::Tensor& grad_,
    const at::Tensor& state_sum_,
    const at::Tensor& param2_,
    int64_t step,
    double learning_rate,
    double weight_decay,
    double lr_decay,
    double eps) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::adagrad_fused_step", std::vector<c10::IValue>({}));
#endif
  TORCH_CHECK(
      learning_rate >= 0, "Expect learning rate >= 0.0, got ", learning_rate);
  TORCH_CHECK(lr_decay >= 0, "Expect lr_decay >=0.0 , got ", lr_decay);
  TORCH_CHECK(eps >= 0, "Expect eps >= 0.0, got ", eps);
  TORCH_CHECK(
      weight_decay >= 0, "Expect weight_decay >= 0.0, got ", weight_decay);

  TORCH_CHECK(
      param_.sizes() == grad_.sizes(),
      "Expect param and grad_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; grad_ sizes: ",
      grad_.sizes());
  TORCH_CHECK(
      param_.sizes() == state_sum_.sizes(),
      "Expect param and state_sum have the same sizes, param sizes: ",
      param_.sizes(),
      "; state_sum sizes: ",
      state_sum_.sizes());
  TORCH_CHECK(
      param2_.numel() == 0 || param_.sizes() == param2_.sizes(),
      "Expect param and param2_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; param2_ sizes: ",
      param2_.sizes());

  auto param = param_.contiguous();
  auto grad = grad_.contiguous();
  auto state_sum = state_sum_.contiguous();
  auto param2 = param2_.contiguous();

  auto grad_dtype = grad_.scalar_type();
  auto param_dtype = param_.scalar_type();
  if (at::ScalarType::Float == grad_dtype) {
    adagrad_fused_step_kernel<float, float>(
        param,
        grad,
        state_sum,
        param2,
        step,
        learning_rate,
        weight_decay,
        lr_decay,
        eps);
  } else if (at::ScalarType::Double == grad_dtype) {
    adagrad_fused_step_kernel<double, double>(
        param,
        grad,
        state_sum,
        param2,
        step,
        learning_rate,
        weight_decay,
        lr_decay,
        eps);
  } else if (
      at::ScalarType::BFloat16 == grad_dtype &&
      at::ScalarType::BFloat16 == param_dtype) {
    adagrad_fused_step_kernel<at::BFloat16, at::BFloat16>(
        param,
        grad,
        state_sum,
        param2,
        step,
        learning_rate,
        weight_decay,
        lr_decay,
        eps);
  } else if (
      at::ScalarType::BFloat16 == grad_dtype &&
      at::ScalarType::Float == param_dtype) {
    adagrad_fused_step_kernel<float, at::BFloat16>(
        param,
        grad,
        state_sum,
        param2,
        step,
        learning_rate,
        weight_decay,
        lr_decay,
        eps);
  } else {
    TORCH_CHECK(false, "expect bfloat16 or float or double param");
  }

  if (!param_.is_contiguous()) {
    param_.copy_(param);
  }
  if (!state_sum_.is_contiguous()) {
    state_sum_.copy_(state_sum);
  }
  if (!param2_.is_contiguous()) {
    param2_.copy_(param2);
  }

  return std::make_tuple(param_, state_sum_);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "adagrad_fused_step(Tensor(a!) param, Tensor grad, Tensor(b!) "
      "state_sum, Tensor trail, int step, float lr, float weight_decay, "
      "float lr_decay, float eps) -> (Tensor(a!), Tensor(b!))",
      torch_ipex::cpu::adagrad_fused_step);
}

} // namespace
