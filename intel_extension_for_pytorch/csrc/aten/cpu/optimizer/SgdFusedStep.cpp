#include "csrc/cpu/vec512/bf16/vec/vec_type_cvt.h"
#include "optimizer.h"

#include <torch/csrc/autograd/function.h>
#include <torch/extension.h>

namespace torch_ipex {
namespace cpu {

using namespace at::vec;

template <typename scalar_t, typename grad_t>
void sgd_fused_step_kernel(
    at::Tensor& param,
    const at::Tensor& grad,
    at::Tensor& momentum_buf,
    at::Tensor& param2,
    double momentum,
    double learning_rate,
    double weight_decay,
    double dampening,
    bool nesterov) {
  scalar_t* param_data = param.data_ptr<scalar_t>();
  scalar_t* grad_data = grad.data_ptr<scalar_t>();
  scalar_t* momentum_buf_data = momentum_buf.data_ptr<scalar_t>();

  using Vec = at::vec::Vectorized<scalar_t>;

  int64_t grain_size = 512;

  // purely element-wise operations
  at::parallel_for(
      0, param.numel(), grain_size, [&](int64_t begin, int64_t end) {
        // local pointers
        scalar_t* param_ptr = param_data + begin;
        scalar_t* grad_ptr = grad_data + begin;
        scalar_t* momentum_buf_ptr = momentum_buf_data + begin;

        const int64_t size = end - begin;
        scalar_t grad_decay = 1 - dampening;
        int64_t d = 0;
        for (; d < size - (size % Vec::size()); d += Vec::size()) {
          Vec param_vec = Vec::loadu(param_ptr + d);
          Vec grad_vec = Vec::loadu(grad_ptr + d) +
              param_vec * Vec(scalar_t(weight_decay));

          if (momentum != 0) {
            Vec momentum_vec =
                Vec::loadu(momentum_buf_ptr + d) * Vec(scalar_t(momentum)) +
                grad_vec * Vec(grad_decay);
            momentum_vec.store(momentum_buf_ptr + d);
            if (nesterov) {
              grad_vec += momentum_vec * Vec(scalar_t(momentum));
            } else {
              grad_vec = momentum_vec;
            }
          }
          param_vec -= grad_vec * Vec(scalar_t(learning_rate));
          param_vec.store(param_ptr + d);
        }
        for (; d < size; d++) {
          scalar_t grad_val = grad_ptr[d] + param_ptr[d] * weight_decay;
          if (momentum != 0) {
            momentum_buf_ptr[d] =
                momentum_buf_ptr[d] * momentum + grad_val * grad_decay;
            if (nesterov) {
              grad_val += momentum_buf_ptr[d] * momentum;
            } else {
              grad_val = momentum_buf_ptr[d];
            }
          }
          param_ptr[d] -= grad_val * learning_rate;
        }
      });
}

template <>
void sgd_fused_step_kernel<at::BFloat16, at::BFloat16>(
    at::Tensor& param,
    const at::Tensor& grad,
    at::Tensor& momentum_buf,
    at::Tensor& param2,
    double momentum,
    double learning_rate,
    double weight_decay,
    double dampening,
    bool nesterov) {
  TORCH_CHECK(
      param.scalar_type() == at::kBFloat16,
      "sgd_fused_step_kernel: expect param to be at::BFloat16");
  TORCH_CHECK(
      grad.scalar_type() == at::kBFloat16,
      "sgd_fused_step_kernel: expect grad to be at::BFloat16");
  TORCH_CHECK(
      momentum_buf.scalar_type() == at::kFloat,
      "sgd_fused_step_kernel: expect stats_sum to be float32");
  TORCH_CHECK(
      param2.scalar_type() == at::kBFloat16,
      "sgd_fused_step_kernel: expect param2 to be at::BFloat16");

  at::BFloat16* param_data = param.data_ptr<at::BFloat16>();
  at::BFloat16* grad_data = grad.data_ptr<at::BFloat16>();
  float* momentum_buf_data = momentum_buf.data_ptr<float>();
  at::BFloat16* param2_data = param2.data_ptr<at::BFloat16>();

  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;

  int64_t grain_size = 512;

  // purely element-wise operations
  at::parallel_for(
      0, param.numel(), grain_size, [&](int64_t begin, int64_t end) {
        // local pointers
        at::BFloat16* param_ptr = param_data + begin;
        at::BFloat16* grad_ptr = grad_data + begin;
        float* momentum_buf_ptr = momentum_buf_data + begin;
        at::BFloat16* param2_ptr = param2_data + begin;

        const int64_t size = end - begin;
        float grad_decay = 1 - dampening;
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

          if (momentum != 0) {
            fVec momentum_vec =
                fVec::loadu(momentum_buf_ptr + d) * fVec(float(momentum)) +
                grad_fvec * fVec(grad_decay);
            fVec momentum_vec2 =
                fVec::loadu(momentum_buf_ptr + d + fVec::size()) *
                    fVec(float(momentum)) +
                grad_fvec2 * fVec(grad_decay);
            momentum_vec.store(momentum_buf_ptr + d);
            momentum_vec2.store(momentum_buf_ptr + d + fVec::size());
            if (nesterov) {
              grad_fvec += momentum_vec * fVec(momentum);
              grad_fvec2 += momentum_vec2 * fVec(momentum);
            } else {
              grad_fvec = momentum_vec;
              grad_fvec2 = momentum_vec2;
            }
          }

          param_fvec -= grad_fvec * fVec(learning_rate);
          param_fvec2 -= grad_fvec2 * fVec(learning_rate);

          std::tie(param_bvec, param2_bvec) =
              bf16::unpack_float_bfloat16(param_fvec, param_fvec2);
          param_bvec.store(param_ptr + d);
          param2_bvec.store(param2_ptr + d);
        }
        for (; d < size; d++) {
          float param_val =
              bf16::pack_bfloat16_float(param_ptr[d], param2_ptr[d]);
          float grad_val = float(grad_ptr[d]) + param_val * weight_decay;
          if (momentum != 0) {
            momentum_buf_ptr[d] =
                momentum_buf_ptr[d] * momentum + grad_val * grad_decay;
            if (nesterov) {
              grad_val += momentum_buf_ptr[d] * momentum;
            } else {
              grad_val = momentum_buf_ptr[d];
            }
          }
          param_val -= grad_val * learning_rate;
          std::tie(param_ptr[d], param2_ptr[d]) =
              bf16::unpack_float_bfloat16(param_val);
        }
      });
}

template <>
void sgd_fused_step_kernel<float, at::BFloat16>(
    at::Tensor& param,
    const at::Tensor& grad,
    at::Tensor& momentum_buf,
    at::Tensor& param2,
    double momentum,
    double learning_rate,
    double weight_decay,
    double dampening,
    bool nesterov) {
  TORCH_CHECK(
      param.scalar_type() == at::kFloat,
      "sgd_fused_step_kernel: expect param to be at::kFloat");
  TORCH_CHECK(
      grad.scalar_type() == at::kBFloat16,
      "sgd_fused_step_kernel: expect grad to be at::BFloat16");
  TORCH_CHECK(
      momentum_buf.scalar_type() == at::kFloat,
      "sgd_fused_step_kernel: expect stats_sum to be float32");
  TORCH_CHECK(
      param2.scalar_type() == at::kBFloat16,
      "sgd_fused_step_kernel: expect param to be at::kBFloat16");

  float* param_data = param.data_ptr<float>();
  at::BFloat16* grad_data = grad.data_ptr<at::BFloat16>();
  float* momentum_buf_data = momentum_buf.data_ptr<float>();
  at::BFloat16* param2_data = param2.data_ptr<at::BFloat16>();

  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;

  int64_t grain_size = 512;

  // purely element-wise operations
  at::parallel_for(
      0, param.numel(), grain_size, [&](int64_t begin, int64_t end) {
        // local pointers
        float* param_ptr = param_data + begin;
        at::BFloat16* grad_ptr = grad_data + begin;
        float* momentum_buf_ptr = momentum_buf_data + begin;
        at::BFloat16* param2_ptr = param2_data + begin;

        const int64_t size = end - begin;
        float grad_decay = 1 - dampening;
        int64_t d = 0;
        for (; d < size - (size % bVec::size()); d += bVec::size()) {
          fVec param_fvec = fVec::loadu(param_ptr + d);
          fVec param_fvec2 = fVec::loadu(param_ptr + d + fVec::size());
          bVec grad_bvec = bVec::loadu(grad_ptr + d);
          fVec grad_fvec, grad_fvec2;
          std::tie(grad_fvec, grad_fvec2) = convert_bfloat16_float(grad_bvec);

          grad_fvec = grad_fvec + param_fvec * fVec(float(weight_decay));
          grad_fvec2 = grad_fvec2 + param_fvec2 * fVec(float(weight_decay));

          if (momentum != 0) {
            fVec momentum_vec =
                fVec::loadu(momentum_buf_ptr + d) * fVec(float(momentum)) +
                grad_fvec * fVec(grad_decay);
            fVec momentum_vec2 =
                fVec::loadu(momentum_buf_ptr + d + fVec::size()) *
                    fVec(float(momentum)) +
                grad_fvec2 * fVec(grad_decay);
            momentum_vec.store(momentum_buf_ptr + d);
            momentum_vec2.store(momentum_buf_ptr + d + fVec::size());
            if (nesterov) {
              grad_fvec += momentum_vec * fVec(momentum);
              grad_fvec2 += momentum_vec2 * fVec(momentum);
            } else {
              grad_fvec = momentum_vec;
              grad_fvec2 = momentum_vec2;
            }
          }

          param_fvec -= grad_fvec * fVec(learning_rate);
          param_fvec2 -= grad_fvec2 * fVec(learning_rate);

          param_fvec.store(param_ptr + d);
          param_fvec2.store(param_ptr + d + fVec::size());
          // sync float param to bfloat16
          bVec param2_bvec = convert_float_bfloat16(param_fvec, param_fvec2);
          param2_bvec.store(param2_ptr + d);
        }
        for (; d < size; d++) {
          float param_val = param_ptr[d];
          float grad_val = float(grad_ptr[d]) + param_val * weight_decay;
          if (momentum != 0) {
            momentum_buf_ptr[d] =
                momentum_buf_ptr[d] * momentum + grad_val * grad_decay;
            if (nesterov) {
              grad_val += momentum_buf_ptr[d] * momentum;
            } else {
              grad_val = momentum_buf_ptr[d];
            }
          }
          param_val -= grad_val * learning_rate;
          param_ptr[d] = param_val;
          param2_ptr[d] = at::BFloat16(param_val);
        }
      });
}

/**
 * SGD fused update kernel.
 * Support Double, Float, BFloat16 training
 *@param param_ Parameters to be update
 *@param grad_ Grad used to update Parameters
 *@param momentum_buf_ momentum to accelerate convergence
 *@param param2_ Used for BF16 training, if param_ is float, param2_ is bf16
 *params need to be synced after update if param_ is BFloat16, param2_ is
 *params_ last 16 bit matissa to construct float params
 *@param momentum Args for momentum.
 *@param learning_rate  Weight for grad while update.
 *@param weight_decay Args for regularization to avoid over-fit.
 *@param dampening Attribute for momentum.
 *@param nesterov Attribute for momentum.
 */
void sgd_fused_step(
    at::Tensor& param_,
    const at::Tensor& grad_,
    at::Tensor& momentum_buf_,
    at::Tensor& param2_,
    double momentum,
    double learning_rate,
    double weight_decay,
    double dampening,
    bool nesterov) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("torch_ipex::sgd_fused_step", std::vector<c10::IValue>({}));
#endif
  TORCH_CHECK(
      learning_rate >= 0, "Expect learning rate >= 0.0, got ", learning_rate);
  TORCH_CHECK(
      weight_decay >= 0, "Expect weight_decay >= 0.0, got ", weight_decay);

  TORCH_CHECK(
      param_.sizes() == grad_.sizes(),
      "Expect param and grad_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; grad_ sizes: ",
      grad_.sizes());
  TORCH_CHECK(
      param_.sizes() == momentum_buf_.sizes(),
      "Expect param and momentum_buf have the same sizes, param sizes: ",
      param_.sizes(),
      "; momentum_buf sizes: ",
      momentum_buf_.sizes());
  TORCH_CHECK(
      param2_.numel() == 0 || param_.sizes() == param2_.sizes(),
      "Expect param and param2_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; param2_ sizes: ",
      param2_.sizes());

  auto param = param_.contiguous();
  auto grad = grad_.contiguous();
  auto momentum_buf = momentum_buf_.contiguous();
  auto param2 = param2_.contiguous();

  auto grad_dtype = grad_.scalar_type();
  auto param_dtype = param_.scalar_type();
  if (at::ScalarType::Float == grad_dtype) {
    sgd_fused_step_kernel<float, float>(
        param,
        grad,
        momentum_buf,
        param2,
        momentum,
        learning_rate,
        weight_decay,
        dampening,
        nesterov);
  } else if (at::ScalarType::Double == grad_dtype) {
    sgd_fused_step_kernel<double, double>(
        param,
        grad,
        momentum_buf,
        param2,
        momentum,
        learning_rate,
        weight_decay,
        dampening,
        nesterov);
  } else if (
      at::ScalarType::BFloat16 == grad_dtype &&
      at::ScalarType::BFloat16 == param_dtype) {
    sgd_fused_step_kernel<at::BFloat16, at::BFloat16>(
        param,
        grad,
        momentum_buf,
        param2,
        momentum,
        learning_rate,
        weight_decay,
        dampening,
        nesterov);
  } else if (
      at::ScalarType::BFloat16 == grad_dtype &&
      at::ScalarType::Float == param_dtype) {
    sgd_fused_step_kernel<float, at::BFloat16>(
        param,
        grad,
        momentum_buf,
        param2,
        momentum,
        learning_rate,
        weight_decay,
        dampening,
        nesterov);
  } else {
    TORCH_CHECK(false, "expect bfloat16 or float or double param");
  }

  if (!param_.is_contiguous()) {
    param_.copy_(param);
  }
  if (!momentum_buf_.is_contiguous()) {
    momentum_buf_.copy_(momentum_buf);
  }
  if (!param2_.is_contiguous()) {
    param2_.copy_(param2);
  }

  return;
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "sgd_fused_step(Tensor param, Tensor grad, Tensor momentum_buf, Tensor "
      "trail, float momentum, float learning_rate, float weight_decay, float "
      "dampening, bool nesterov) -> ()",
      torch_ipex::cpu::sgd_fused_step);
}

} // namespace
