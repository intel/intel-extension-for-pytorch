#include <aten/optimizer/optimizer.h>
#include "vec/vec.h"

#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

namespace {

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
    bool nesterov,
    bool momentum_buf_initialized) {
  scalar_t* param_data = param.data_ptr<scalar_t>();
  scalar_t* grad_data = grad.data_ptr<scalar_t>();
  scalar_t* momentum_buf_data =
      momentum_buf.defined() ? momentum_buf.data_ptr<scalar_t>() : nullptr;

  using Vec = at::vec::Vectorized<scalar_t>;

  int64_t grain_size = 512;
  scalar_t grad_decay_val = 1.0 - dampening;
  scalar_t weight_decay_val = scalar_t(weight_decay);
  scalar_t momentum_val = scalar_t(momentum);
  scalar_t learning_rate_val = scalar_t(learning_rate);
  // purely element-wise operations
  at::parallel_for(
      0, param.numel(), grain_size, [&](int64_t begin, int64_t end) {
        // local pointers
        scalar_t* param_ptr = param_data + begin;
        scalar_t* grad_ptr = grad_data + begin;
        scalar_t* momentum_buf_ptr = momentum_buf_data + begin;

        const int64_t size = end - begin;
        int64_t d = 0;
        for (; d < size - (size % Vec::size()); d += Vec::size()) {
          Vec param_vec = Vec::loadu(param_ptr + d);
          Vec grad_vec =
              Vec::loadu(grad_ptr + d) + param_vec * Vec(weight_decay_val);

          if (momentum != 0) {
            Vec momentum_vec;
            if (!momentum_buf_initialized) {
              momentum_vec = grad_vec;
            } else {
              momentum_vec =
                  Vec::loadu(momentum_buf_ptr + d) * Vec(momentum_val) +
                  grad_vec * Vec(grad_decay_val);
            }
            momentum_vec.store(momentum_buf_ptr + d);
            if (nesterov) {
              grad_vec += momentum_vec * Vec(momentum_val);
            } else {
              grad_vec = momentum_vec;
            }
          }
          param_vec -= grad_vec * Vec(learning_rate_val);
          param_vec.store(param_ptr + d);
        }
        for (; d < size; d++) {
          scalar_t grad_val = grad_ptr[d] + param_ptr[d] * weight_decay_val;
          if (momentum != 0) {
            if (!momentum_buf_initialized) {
              momentum_buf_ptr[d] = grad_val;
            } else {
              momentum_buf_ptr[d] = momentum_buf_ptr[d] * momentum_val +
                  grad_val * grad_decay_val;
            }
            if (nesterov) {
              grad_val += momentum_buf_ptr[d] * momentum_val;
            } else {
              grad_val = momentum_buf_ptr[d];
            }
          }
          param_ptr[d] -= grad_val * learning_rate_val;
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
    bool nesterov,
    bool momentum_buf_initialized) {
  TORCH_CHECK(
      param.scalar_type() == at::kBFloat16,
      "sgd_fused_step_kernel: expect param to be at::BFloat16");
  TORCH_CHECK(
      grad.scalar_type() == at::kBFloat16,
      "sgd_fused_step_kernel: expect grad to be at::BFloat16");
  TORCH_CHECK(
      !momentum_buf.defined() || momentum_buf.scalar_type() == at::kFloat,
      "sgd_fused_step_kernel: expect momentum_buf to be float32");
  TORCH_CHECK(
      param2.scalar_type() == at::kBFloat16,
      "sgd_fused_step_kernel: expect param2 to be at::BFloat16");

  at::BFloat16* param_data = param.data_ptr<at::BFloat16>();
  at::BFloat16* grad_data = grad.data_ptr<at::BFloat16>();
  float* momentum_buf_data =
      momentum_buf.defined() ? momentum_buf.data_ptr<float>() : nullptr;
  at::BFloat16* param2_data = param2.data_ptr<at::BFloat16>();

  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;

  int64_t grain_size = 512;
  float grad_decay_val = 1 - dampening;
  float weight_decay_val = float(weight_decay);
  float momentum_val = float(momentum);
  float learning_rate_val = float(learning_rate);
  // purely element-wise operations
  at::parallel_for(
      0, param.numel(), grain_size, [&](int64_t begin, int64_t end) {
        // local pointers
        at::BFloat16* param_ptr = param_data + begin;
        at::BFloat16* grad_ptr = grad_data + begin;
        float* momentum_buf_ptr = momentum_buf_data + begin;
        at::BFloat16* param2_ptr = param2_data + begin;

        const int64_t size = end - begin;
        int64_t d = 0;
        for (; d < size - (size % bVec::size()); d += bVec::size()) {
          bVec param_bvec = bVec::loadu(param_ptr + d);
          bVec param2_bvec = bVec::loadu(param2_ptr + d);
          fVec param_fvec, param_fvec2;
          std::tie(param_fvec, param_fvec2) =
              at::vec::pack_bfloat16_float(param_bvec, param2_bvec);

          bVec grad_bvec = bVec::loadu(grad_ptr + d);
          fVec grad_fvec, grad_fvec2;
          std::tie(grad_fvec, grad_fvec2) = convert_bfloat16_float(grad_bvec);

          grad_fvec = grad_fvec + param_fvec * fVec(weight_decay_val);
          grad_fvec2 = grad_fvec2 + param_fvec2 * fVec(weight_decay_val);

          if (momentum != 0) {
            fVec momentum_vec, momentum_vec2;
            if (!momentum_buf_initialized) {
              momentum_vec = grad_fvec;
              momentum_vec2 = grad_fvec2;
            } else {
              momentum_vec =
                  fVec::loadu(momentum_buf_ptr + d) * fVec(momentum_val) +
                  grad_fvec * fVec(grad_decay_val);
              momentum_vec2 = fVec::loadu(momentum_buf_ptr + d + fVec::size()) *
                      fVec(momentum_val) +
                  grad_fvec2 * fVec(grad_decay_val);
            }
            momentum_vec.store(momentum_buf_ptr + d);
            momentum_vec2.store(momentum_buf_ptr + d + fVec::size());
            if (nesterov) {
              grad_fvec += momentum_vec * fVec(momentum_val);
              grad_fvec2 += momentum_vec2 * fVec(momentum_val);
            } else {
              grad_fvec = momentum_vec;
              grad_fvec2 = momentum_vec2;
            }
          }

          param_fvec -= grad_fvec * fVec(learning_rate_val);
          param_fvec2 -= grad_fvec2 * fVec(learning_rate_val);

          std::tie(param_bvec, param2_bvec) =
              at::vec::unpack_float_bfloat16(param_fvec, param_fvec2);
          param_bvec.store(param_ptr + d);
          param2_bvec.store(param2_ptr + d);
        }
        for (; d < size; d++) {
          float param_val =
              at::vec::pack_bfloat16_float(param_ptr[d], param2_ptr[d]);
          float grad_val = float(grad_ptr[d]) + param_val * weight_decay_val;
          if (momentum != 0) {
            if (!momentum_buf_initialized) {
              momentum_buf_ptr[d] = grad_val;
            } else {
              momentum_buf_ptr[d] = momentum_buf_ptr[d] * momentum_val +
                  grad_val * grad_decay_val;
            }
            if (nesterov) {
              grad_val += momentum_buf_ptr[d] * momentum_val;
            } else {
              grad_val = momentum_buf_ptr[d];
            }
          }
          param_val -= grad_val * learning_rate_val;
          std::tie(param_ptr[d], param2_ptr[d]) =
              at::vec::unpack_float_bfloat16(param_val);
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
    bool nesterov,
    bool momentum_buf_initialized) {
  TORCH_CHECK(
      param.scalar_type() == at::kFloat,
      "sgd_fused_step_kernel: expect param to be at::kFloat");
  TORCH_CHECK(
      grad.scalar_type() == at::kBFloat16,
      "sgd_fused_step_kernel: expect grad to be at::BFloat16");
  TORCH_CHECK(
      !momentum_buf.defined() || momentum_buf.scalar_type() == at::kFloat,
      "sgd_fused_step_kernel: expect momentum_buf to be float32");
  TORCH_CHECK(
      param2.scalar_type() == at::kBFloat16,
      "sgd_fused_step_kernel: expect param to be at::kBFloat16");

  float* param_data = param.data_ptr<float>();
  at::BFloat16* grad_data = grad.data_ptr<at::BFloat16>();
  float* momentum_buf_data =
      momentum_buf.defined() ? momentum_buf.data_ptr<float>() : nullptr;
  at::BFloat16* param2_data = param2.data_ptr<at::BFloat16>();

  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;

  int64_t grain_size = 512;
  float grad_decay_val = 1 - dampening;
  float weight_decay_val = float(weight_decay);
  float momentum_val = float(momentum);
  float learning_rate_val = float(learning_rate);
  // purely element-wise operations
  at::parallel_for(
      0, param.numel(), grain_size, [&](int64_t begin, int64_t end) {
        // local pointers
        float* param_ptr = param_data + begin;
        at::BFloat16* grad_ptr = grad_data + begin;
        float* momentum_buf_ptr = momentum_buf_data + begin;
        at::BFloat16* param2_ptr = param2_data + begin;

        const int64_t size = end - begin;
        int64_t d = 0;
        for (; d < size - (size % bVec::size()); d += bVec::size()) {
          fVec param_fvec = fVec::loadu(param_ptr + d);
          fVec param_fvec2 = fVec::loadu(param_ptr + d + fVec::size());
          bVec grad_bvec = bVec::loadu(grad_ptr + d);
          fVec grad_fvec, grad_fvec2;
          std::tie(grad_fvec, grad_fvec2) = convert_bfloat16_float(grad_bvec);

          grad_fvec = grad_fvec + param_fvec * fVec(weight_decay_val);
          grad_fvec2 = grad_fvec2 + param_fvec2 * fVec(weight_decay_val);

          if (momentum != 0) {
            fVec momentum_vec, momentum_vec2;
            if (!momentum_buf_initialized) {
              momentum_vec = grad_fvec;
              momentum_vec2 = grad_fvec2;
            } else {
              momentum_vec =
                  fVec::loadu(momentum_buf_ptr + d) * fVec(momentum_val) +
                  grad_fvec * fVec(grad_decay_val);
              momentum_vec2 = fVec::loadu(momentum_buf_ptr + d + fVec::size()) *
                      fVec(momentum_val) +
                  grad_fvec2 * fVec(grad_decay_val);
            }
            momentum_vec.store(momentum_buf_ptr + d);
            momentum_vec2.store(momentum_buf_ptr + d + fVec::size());
            if (nesterov) {
              grad_fvec += momentum_vec * fVec(momentum_val);
              grad_fvec2 += momentum_vec2 * fVec(momentum_val);
            } else {
              grad_fvec = momentum_vec;
              grad_fvec2 = momentum_vec2;
            }
          }

          param_fvec -= grad_fvec * fVec(learning_rate_val);
          param_fvec2 -= grad_fvec2 * fVec(learning_rate_val);

          param_fvec.store(param_ptr + d);
          param_fvec2.store(param_ptr + d + fVec::size());
          // sync float param to bfloat16
          bVec param2_bvec = convert_float_bfloat16(param_fvec, param_fvec2);
          param2_bvec.store(param2_ptr + d);
        }
        for (; d < size; d++) {
          float param_val = param_ptr[d];
          float grad_val = float(grad_ptr[d]) + param_val * weight_decay_val;
          if (momentum != 0) {
            if (!momentum_buf_initialized) {
              momentum_buf_ptr[d] = grad_val;
            } else {
              momentum_buf_ptr[d] = momentum_buf_ptr[d] * momentum_val +
                  grad_val * grad_decay_val;
            }
            if (nesterov) {
              grad_val += momentum_buf_ptr[d] * momentum_val;
            } else {
              grad_val = momentum_buf_ptr[d];
            }
          }
          param_val -= grad_val * learning_rate_val;
          param_ptr[d] = param_val;
          param2_ptr[d] = at::BFloat16(param_val);
        }
      });
}

c10::optional<at::Tensor> sgd_fused_step_kernel_impl(
    at::Tensor& param_,
    const at::Tensor& grad_,
    const c10::optional<at::Tensor>& momentum_buf_,
    at::Tensor& param2_,
    double momentum,
    double learning_rate,
    double weight_decay,
    double dampening,
    bool nesterov) {
  auto param = param_.contiguous();
  auto grad = grad_.contiguous();
  auto param2 = param2_.contiguous();

  at::Tensor momentum_buf;
  bool momentum_buf_initialized;
  if (momentum != 0) {
    if (!momentum_buf_.has_value()) {
      auto acc_dtype =
          param.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
      momentum_buf = at::empty_like(param, acc_dtype);
      momentum_buf_initialized = false;
    } else {
      momentum_buf = momentum_buf_.value().contiguous();
      momentum_buf_initialized = true;
    }
  }

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
        nesterov,
        momentum_buf_initialized);
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
        nesterov,
        momentum_buf_initialized);
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
        nesterov,
        momentum_buf_initialized);
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
        nesterov,
        momentum_buf_initialized);
  } else {
    TORCH_CHECK(false, "expect bfloat16 or float or double param");
  }
  if (!param_.is_contiguous()) {
    param_.copy_(param);
  }

  if (!param2_.is_contiguous()) {
    param2_.copy_(param2);
  }

  if (momentum_buf_.has_value() && !momentum_buf_.value().is_contiguous()) {
    momentum_buf_.value().copy_(momentum_buf);
  }

  if (momentum == 0) {
    return c10::nullopt;
  } else
    return momentum_buf;
}

} // anonymous namespace

REGISTER_DISPATCH(sgd_fused_step_kernel_stub, &sgd_fused_step_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
