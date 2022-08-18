#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/record_function.h>

#include <core/Memory.h>
#include <core/TensorImplUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"

#include "comm/ATDispatch.h"
#include "comm/Numerics.h"

#include <aten/operators/MemoryAccess.h>

using namespace xpu::dpcpp;
using namespace xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <
    int vec_size,
    typename vec_grad_t,
    typename elem_grad_t,
    typename vec_w_t,
    typename elem_w_t,
    typename scalar_t,
    typename accscalar_t,
    typename vec_mw_t,
    typename elem_mw_t>
void vec_kernel_AdamW(
    float* master_weight_ptr,
    scalar_t* weight_ptr,
    scalar_t* grad_ptr,
    float* exp_avg_ptr,
    float* exp_avg_sq_ptr,
    float* max_exp_avg_sq_ptr,
    int64_t step,
    double lr,
    double eps,
    double beta1,
    double beta2,
    double weight_decay,
    const bool amsgrad,
    const bool transformer,
    const bool correct_bias,
    int64_t total_element,
    sycl::item<1> item) {
  auto id = item.get_id(0);

  // cast grad, weight and other memory using vector
  vec_w_t* weight_vec = reinterpret_cast<vec_w_t*>(weight_ptr);
  vec_grad_t* grad_vec = reinterpret_cast<vec_grad_t*>(grad_ptr);
  vec_mw_t* master_weight_vec = reinterpret_cast<vec_mw_t*>(master_weight_ptr);
  vec_mw_t* exp_avg_vec = reinterpret_cast<vec_mw_t*>(exp_avg_ptr);
  vec_mw_t* exp_avg_sq_vec = reinterpret_cast<vec_mw_t*>(exp_avg_sq_ptr);

  // if amsgrad is false, max_exp_avg_sq_vec is nullptr
  vec_mw_t* max_exp_avg_sq_vec =
      amsgrad ? reinterpret_cast<vec_mw_t*>(max_exp_avg_sq_ptr) : nullptr;

  auto remaining = total_element - id * vec_size;

  // for handling remaining tail
  if (remaining < vec_size) {
    for (auto v_index = 0; v_index < remaining; ++v_index) {
      // kick out tail
      auto linear_idx = id * vec_size + v_index;
      // master weight grad should be fp32 to involve in computation to keep
      // acc.
      auto grad_elm = static_cast<accscalar_t>(grad_ptr[linear_idx]);

      // official torch
      // TODO: optimize ILP
      auto master_weight_elem = master_weight_ptr[linear_idx];
      if (!correct_bias) {
        master_weight_elem =
            master_weight_elem - master_weight_elem * (lr * weight_decay);
      }

      // official torch
      float bias_correction1 = 0.0;
      float bias_correction2 = 0.0;
      if ((!transformer) || (correct_bias)) {
        bias_correction1 = 1.0 - Numerics<float>::pow(beta1, step);
        bias_correction2 = 1.0 - Numerics<float>::pow(beta2, step);
      }

      // official torch
      // exp_avg
      auto exp_avg_ele = exp_avg_ptr[linear_idx];
      exp_avg_ele = exp_avg_ele * beta1 + grad_elm * (1.0 - beta1);
      exp_avg_ptr[linear_idx] = exp_avg_ele;

      // exp_avg_sq
      auto exp_avg_sq_ele = exp_avg_sq_ptr[linear_idx];
      exp_avg_sq_ele =
          exp_avg_sq_ele * beta2 + grad_elm * grad_elm * (1.0 - beta2);
      exp_avg_sq_ptr[linear_idx] = exp_avg_sq_ele;

      // amsgrad
      float denom = 0.0;
      if (amsgrad) {
        // official torch
        max_exp_avg_sq_ptr[linear_idx] =
            max_exp_avg_sq_ptr[linear_idx] < exp_avg_sq_ele
            ? exp_avg_sq_ele
            : max_exp_avg_sq_ptr[linear_idx];
        denom = Numerics<float>::sqrt(
                    max_exp_avg_sq_ptr[linear_idx] / bias_correction2) +
            eps;
      } else if (!transformer) {
        // official torch
        denom = Numerics<float>::sqrt(exp_avg_sq_ele / bias_correction2) + eps;
      } else {
        denom = Numerics<float>::sqrt(exp_avg_sq_ele) + eps;
      }

      float step_size = static_cast<float>(lr);
      if (!transformer) {
        // official torch
        step_size = lr / bias_correction1;
      } else if (correct_bias) {
        step_size = step_size * Numerics<float>::sqrt(bias_correction2) /
            bias_correction1;
      }

      // official torch
      master_weight_elem =
          master_weight_elem - step_size * (exp_avg_ele / denom);

      if (transformer && (weight_decay > 0.0)) {
        master_weight_elem =
            master_weight_elem - master_weight_elem * (lr * weight_decay);
      }

      // update master weight fp32
      master_weight_ptr[linear_idx] = master_weight_elem;

      // update real weight bf16/fp16
      weight_ptr[linear_idx] = scalar_t(master_weight_elem);
    }
  } else {
    // vector read
    vec_grad_t grad_value = grad_vec[id];
    vec_mw_t exp_avg_value = exp_avg_vec[id];
    vec_mw_t exp_avg_sq_value = exp_avg_sq_vec[id];
    vec_mw_t master_weight_value = master_weight_vec[id];

    // for vector write back
    vec_w_t temp_weight;
    vec_mw_t temp_master_weight;
    vec_mw_t temp_exp_avg;
    vec_mw_t temp_exp_avg_sq;

#pragma unroll
    for (auto v_index = 0; v_index < vec_size; ++v_index) {
      // master weight grad should be fp32 to involve in computation to keep
      // acc.
      auto grad_elm = static_cast<accscalar_t>(
          at::native::Memory::detail::bitwise_cast<scalar_t>(
              grad_value[v_index]));

      // official torch
      // TODO: ILP
      auto master_weight_elem = at::native::Memory::detail::bitwise_cast<float>(
          master_weight_value[v_index]);
      if (!transformer) {
        master_weight_elem =
            master_weight_elem - master_weight_elem * (lr * weight_decay);
      }

      // official torch
      float bias_correction1 = 0.0;
      float bias_correction2 = 0.0;
      if ((!transformer) || (correct_bias)) {
        bias_correction1 = 1.0 - Numerics<float>::pow(beta1, step);
        bias_correction2 = 1.0 - Numerics<float>::pow(beta2, step);
      }

      // official torch
      // exp_avg
      auto exp_avg_ele = at::native::Memory::detail::bitwise_cast<float>(
          exp_avg_value[v_index]);
      exp_avg_ele = exp_avg_ele * beta1 + grad_elm * (1.0 - beta1);
      temp_exp_avg[v_index] =
          at::native::Memory::detail::bitwise_cast<elem_mw_t>(exp_avg_ele);

      // exp_avg_sq
      auto exp_avg_sq_ele = at::native::Memory::detail::bitwise_cast<float>(
          exp_avg_sq_value[v_index]);
      exp_avg_sq_ele =
          exp_avg_sq_ele * beta2 + grad_elm * grad_elm * (1.0 - beta2);
      temp_exp_avg_sq[v_index] =
          at::native::Memory::detail::bitwise_cast<elem_mw_t>(exp_avg_sq_ele);

      // amsgrad
      float denom = 0.0;
      if (amsgrad) {
        // official torch
        auto max_exp_avg_sq_ele =
            at::native::Memory::detail::bitwise_cast<float>(
                max_exp_avg_sq_vec[id][v_index]);
        max_exp_avg_sq_ele = max_exp_avg_sq_ele < exp_avg_sq_ele
            ? exp_avg_sq_ele
            : max_exp_avg_sq_ele;
        // max_exp_avg_sq_vec is up to amsgrad, so vector write is not used for
        // it
        max_exp_avg_sq_vec[id][v_index] =
            at::native::Memory::detail::bitwise_cast<elem_mw_t>(
                max_exp_avg_sq_ele);
        denom =
            Numerics<float>::sqrt(max_exp_avg_sq_ele / bias_correction2) + eps;
      } else if (!transformer) {
        // official torch
        denom = Numerics<float>::sqrt(exp_avg_sq_ele / bias_correction2) + eps;
      } else {
        denom = Numerics<float>::sqrt(exp_avg_sq_ele) + eps;
      }

      float step_size = static_cast<float>(lr);
      if (!transformer) {
        // official torch
        step_size = lr / bias_correction1;
      } else if (correct_bias) {
        step_size = step_size * Numerics<float>::sqrt(bias_correction2) /
            bias_correction1;
      }

      // official torch
      master_weight_elem =
          master_weight_elem - step_size * (exp_avg_ele / denom);

      if (transformer && (weight_decay > 0.0)) {
        master_weight_elem =
            master_weight_elem - master_weight_elem * (lr * weight_decay);
      }

      // update master weight fp32
      temp_master_weight[v_index] =
          at::native::Memory::detail::bitwise_cast<elem_mw_t>(
              master_weight_elem);

      // update real weight bf16/fp16
      auto scalar_weigt = scalar_t(master_weight_elem);
      temp_weight[v_index] =
          at::native::Memory::detail::bitwise_cast<elem_w_t>(scalar_weigt);
    }

    // write back
    // update exp_avg
    exp_avg_vec[id] = temp_exp_avg;

    // update exp_avg_sq
    exp_avg_sq_vec[id] = temp_exp_avg_sq;

    // update master weight fp32
    master_weight_vec[id] = temp_master_weight;

    // update real weight bf16/fp16
    weight_vec[id] = temp_weight;
  }
}

// Here is the migrated AdamW kernel,
// Official AdamW:
// https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html.
// transformer AdamW:
// https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py
// Thought: Use vector to read/write BF16/FP16 grad/weight and FP32 master
// weight for memory bandwidth, additional 2 control flags are used.
// transformer: switch the official AdamW and transformer AdamW.
// correct_bias: control the bias calculation behaviour.
template <typename scalar_t>
static void ComputeAdamWKernel(
    Tensor& master_weight,
    Tensor& weight,
    Tensor& grad,
    const bool amsgrad,
    Tensor& avg,
    Tensor& avg_sq,
    Tensor& max_avg_sq,
    int64_t& step,
    double lr,
    double eps,
    double beta1,
    double beta2,
    double weight_decay,
    const bool transformer,
    const bool correct_bias) {
  auto& queue = dpcppGetCurrentQueue();

  auto vec_mw_size = at::native::Memory::can_vectorize_up_to<float>(
      getDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(master_weight.data_ptr<float>()));

  // vector size is chosen as FP32's vector size, because FP32 R/W are much more
  // than Scalar_t
  auto vec_size = vec_mw_size;

  auto total_element = master_weight.numel();

  auto global_range = (total_element % vec_size == 0)
      ? (total_element / vec_size)
      : (total_element / vec_size + 1);

// launch vector kernel for AdamW, code pass according to vector size
#define VEC_ADAMW_KERNEL(vec_size)                                            \
  {                                                                           \
    auto cgf = DPCPP_Q_CGF(cgh) {                                             \
      auto master_weight_ptr = master_weight.data_ptr<float>();               \
      auto weight_ptr = weight.data_ptr<scalar_t>();                          \
      auto grad_ptr = grad.data_ptr<scalar_t>();                              \
      auto exp_avg_ptr = avg.data_ptr<float>();                               \
      auto exp_avg_sq_ptr = avg_sq.data_ptr<float>();                         \
      auto max_exp_avg_sq_ptr =                                               \
          amsgrad ? max_avg_sq.data_ptr<float>() : nullptr;                   \
      using vec_grad_t = typename at::native::Memory::                        \
          aligned_vector<scalar_t, vec_size>::type;                           \
      using elem_grad_t = typename at::native::Memory::                       \
          aligned_vector<scalar_t, vec_size>::element_type;                   \
      using vec_mw_t =                                                        \
          typename at::native::Memory::aligned_vector<float, vec_size>::type; \
      using elem_mw_t = typename at::native::Memory::                         \
          aligned_vector<float, vec_size>::element_type;                      \
      using vec_w_t = vec_grad_t;                                             \
      using elem_w_t = elem_grad_t;                                           \
      using accscalar_t = acc_type<scalar_t>;                                 \
      auto kfn = DPCPP_Q_KFN(sycl::item<1> item) {                            \
        vec_kernel_AdamW<                                                     \
            vec_size,                                                         \
            vec_grad_t,                                                       \
            elem_grad_t,                                                      \
            vec_w_t,                                                          \
            elem_w_t,                                                         \
            scalar_t,                                                         \
            accscalar_t,                                                      \
            vec_mw_t,                                                         \
            elem_mw_t>(                                                       \
            master_weight_ptr,                                                \
            weight_ptr,                                                       \
            grad_ptr,                                                         \
            exp_avg_ptr,                                                      \
            exp_avg_sq_ptr,                                                   \
            max_exp_avg_sq_ptr,                                               \
            step,                                                             \
            lr,                                                               \
            eps,                                                              \
            beta1,                                                            \
            beta2,                                                            \
            weight_decay,                                                     \
            amsgrad,                                                          \
            transformer,                                                      \
            correct_bias,                                                     \
            total_element,                                                    \
            item);                                                            \
      };                                                                      \
      cgh.parallel_for(sycl::range<1>(global_range), kfn);                    \
    };                                                                        \
    DPCPP_Q_SUBMIT(queue, cgf);                                               \
  }

  switch (vec_size) {
    case 16: {
      VEC_ADAMW_KERNEL(16);
      break;
    }
    case 8: {
      VEC_ADAMW_KERNEL(8);
      break;
    }
    case 4: {
      VEC_ADAMW_KERNEL(4);
      break;
    }
    case 2: {
      VEC_ADAMW_KERNEL(2);
      break;
    }
    case 1: {
      VEC_ADAMW_KERNEL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for AdamW kernel. vec size ",
          vec_size);
  }
#undef VEC_ADAMW_KERNEL
}
} // namespace impl

// official torch AdamW
Tensor& fused_adamWMasterWeight(
    Tensor& master_weight,
    Tensor& weight,
    Tensor& grad,
    const bool amsgrad,
    Tensor& avg,
    Tensor& avg_sq,
    Tensor& max_avg_sq,
    int64_t& step,
    double lr,
    double eps,
    double beta1,
    double beta2,
    double weight_decay) {
  RECORD_FUNCTION(
      "fused_adamWMasterWeight",
      std::vector<c10::IValue>(
          {master_weight, weight, grad, amsgrad, avg, avg_sq, max_avg_sq}));

  // support contiguous and channels_last contiguous
  auto memory_format = master_weight.suggest_memory_format();
  master_weight = master_weight.contiguous(memory_format);
  weight = weight.contiguous(memory_format);
  grad = grad.contiguous(memory_format);

  // scalar_t = weight dtype = grad dtype
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weight.scalar_type(),
      "apply_migrated_official_AdamW_dpcpp",
      [&] {
        impl::ComputeAdamWKernel<scalar_t>(
            master_weight,
            weight,
            grad,
            amsgrad,
            avg,
            avg_sq,
            max_avg_sq,
            step,
            lr,
            eps,
            beta1,
            beta2,
            weight_decay,
            /*transformer*/ false,
            /*correct_bias*/ true);
      });
  return master_weight;
}

// transformer AdamW
Tensor& transformer_adamWMasterWeight(
    Tensor& master_weight,
    Tensor& weight,
    Tensor& grad,
    Tensor& avg,
    Tensor& avg_sq,
    Tensor& max_avg_sq,
    int64_t& step,
    double lr,
    double eps,
    double beta1,
    double beta2,
    double weight_decay,
    const bool correct_bias) {
  RECORD_FUNCTION(
      "transformer_adamWMasterWeight",
      std::vector<c10::IValue>(
          {master_weight, weight, grad, avg, avg_sq, max_avg_sq}));

  // support contiguous and channels_last contiguous
  auto memory_format = master_weight.suggest_memory_format();
  master_weight = master_weight.contiguous(memory_format);
  weight = weight.contiguous(memory_format);
  grad = grad.contiguous(memory_format);

  // scalar_t = weight dtype = grad dtype
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weight.scalar_type(),
      "apply_migrated_transformer_AdamW_dpcpp",
      [&] {
        impl::ComputeAdamWKernel<scalar_t>(
            master_weight,
            weight,
            grad,
            /*amsgrad*/ false,
            avg,
            avg_sq,
            max_avg_sq,
            step,
            lr,
            eps,
            beta1,
            beta2,
            weight_decay,
            /*transformer*/ true,
            correct_bias);
      });
  return master_weight;
}

} // namespace AtenIpexTypeXPU
} // namespace at
