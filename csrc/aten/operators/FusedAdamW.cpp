#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/record_function.h>

#include <core/Memory.h>
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

template <int vec_size, typename scalar_t>
void launch_vec_kernel_AdamWMasterWeight(
    Tensor& master_weight,
    Tensor& weight,
    Tensor& grad,
    Tensor& avg,
    Tensor& avg_sq,
    Tensor& max_avg_sq,
    const double step,
    const double lr,
    const double eps,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const bool amsgrad,
    const int64_t total_element,
    const int64_t global_range) {
  auto& queue = dpcppGetCurrentQueue();
  using accscalar_t = acc_type<scalar_t>;

  auto master_weight_ptr = master_weight.data_ptr<float>();
  auto weight_ptr = weight.data_ptr<scalar_t>();
  auto grad_ptr = grad.data_ptr<scalar_t>();
  auto exp_avg_ptr = avg.data_ptr<float>();
  auto exp_avg_sq_ptr = avg_sq.data_ptr<float>();
  auto max_exp_avg_sq_ptr = amsgrad ? max_avg_sq.data_ptr<float>() : nullptr;

  // pre calculate on host side
  auto exp_avg_ele_coefficient = static_cast<float>(1 - beta1);
  auto exp_avg_sq_ele_coefficient = static_cast<float>(1 - beta2);
  auto bias_correction1 =
      static_cast<float>(1.0 - std::pow(static_cast<float>(beta1), step));
  auto bias_correction2 =
      static_cast<float>(1.0 - std::pow(static_cast<float>(beta2), step));
  auto step_size = static_cast<float>(lr / bias_correction1);

  // vec_t is used for vectorization weight and grad
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  // vec_mw_t is used for vectorization master weight, exp, exp_sq and
  // max_exp_sq
  using vec_mw_t = at::native::Memory::aligned_vector_loop<float, vec_size>;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(sycl::range<1>{global_range}, [=](sycl::item<1> item) {
      auto id = item.get_id(0);

      // cast grad, weight and other memory using vector
      vec_t* weight_vec = reinterpret_cast<vec_t*>(weight_ptr);
      vec_mw_t* master_weight_vec =
          reinterpret_cast<vec_mw_t*>(master_weight_ptr);
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

          auto master_weight_elem = master_weight_ptr[linear_idx];
          master_weight_elem =
              master_weight_elem - master_weight_elem * (lr * weight_decay);

          // exp_avg
          auto exp_avg_ele = exp_avg_ptr[linear_idx];
          exp_avg_ele =
              exp_avg_ele * beta1 + grad_elm * exp_avg_ele_coefficient;
          exp_avg_ptr[linear_idx] = static_cast<float>(exp_avg_ele);

          // exp_avg_sq
          auto exp_avg_sq_ele = exp_avg_sq_ptr[linear_idx];
          exp_avg_sq_ele = exp_avg_sq_ele * beta2 +
              grad_elm * grad_elm * exp_avg_sq_ele_coefficient;
          exp_avg_sq_ptr[linear_idx] = static_cast<float>(exp_avg_sq_ele);

          // amsgrad
          if (amsgrad) {
            max_exp_avg_sq_ptr[linear_idx] =
                max_exp_avg_sq_ptr[linear_idx] < exp_avg_sq_ele
                ? static_cast<float>(exp_avg_sq_ele)
                : max_exp_avg_sq_ptr[linear_idx];
            master_weight_elem = master_weight_elem -
                step_size * exp_avg_ele /
                    (Numerics<float>::sqrt(
                         max_exp_avg_sq_ptr[linear_idx] / bias_correction2) +
                     eps);
          } else {
            master_weight_elem = master_weight_elem -
                step_size * exp_avg_ele /
                    (Numerics<float>::sqrt(exp_avg_sq_ele / bias_correction2) +
                     eps);
          }

          // update master weight fp32
          master_weight_ptr[linear_idx] =
              static_cast<float>(master_weight_elem);

          // update real weight bf16/fp16
          weight_ptr[linear_idx] = static_cast<scalar_t>(master_weight_elem);
        }
      } else {
        // vector read
        vec_mw_t exp_avg_value = exp_avg_vec[id];
        vec_mw_t exp_avg_sq_value = exp_avg_sq_vec[id];
        vec_mw_t master_weight_value = master_weight_vec[id];

        // for vector write back
        vec_t temp_weight;
        vec_mw_t temp_master_weight;
        vec_mw_t temp_exp_avg;
        vec_mw_t temp_exp_avg_sq;

#pragma unroll
        for (auto v_index = 0; v_index < vec_size; ++v_index) {
          // master weight grad should be fp32 to involve in computation to keep
          // acc.
          // FIXME: vector read and all grad values are 0, so use point read
          auto grad_elm =
              static_cast<accscalar_t>(grad_ptr[id * vec_size + v_index]);

          auto master_weight_elem = master_weight_value[v_index];
          master_weight_elem =
              master_weight_elem - master_weight_elem * (lr * weight_decay);

          // exp_avg
          auto exp_avg_ele = exp_avg_value[v_index];
          exp_avg_ele =
              exp_avg_ele * beta1 + grad_elm * exp_avg_ele_coefficient;
          temp_exp_avg[v_index] = static_cast<float>(exp_avg_ele);

          // exp_avg_sq
          auto exp_avg_sq_ele = exp_avg_sq_value[v_index];
          exp_avg_sq_ele = exp_avg_sq_ele * beta2 +
              grad_elm * grad_elm * exp_avg_sq_ele_coefficient;
          temp_exp_avg_sq[v_index] = static_cast<float>(exp_avg_sq_ele);

          // amsgrad
          if (amsgrad) {
            auto max_exp_avg_sq_ele = max_exp_avg_sq_vec[id][v_index];
            max_exp_avg_sq_ele = max_exp_avg_sq_ele < exp_avg_sq_ele
                ? exp_avg_sq_ele
                : max_exp_avg_sq_ele;
            max_exp_avg_sq_vec[id][v_index] = max_exp_avg_sq_ele;
            master_weight_elem = master_weight_elem -
                step_size * exp_avg_ele /
                    (Numerics<float>::sqrt(
                         max_exp_avg_sq_ele / bias_correction2) +
                     eps);
          } else {
            master_weight_elem = master_weight_elem -
                step_size * exp_avg_ele /
                    (Numerics<float>::sqrt(exp_avg_sq_ele / bias_correction2) +
                     eps);
          }

          // update master weight fp32
          temp_master_weight[v_index] = static_cast<float>(master_weight_elem);

          // update real weight bf16/fp16
          temp_weight[v_index] = static_cast<scalar_t>(master_weight_elem);
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
    });
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

// Here is the migrated AdamW Kernel
// no master weight, scalar_t = float32
template <int vec_size, typename scalar_t>
void launch_vec_kernel_AdamW(
    Tensor& weight,
    Tensor& grad,
    Tensor& avg,
    Tensor& avg_sq,
    Tensor& max_avg_sq,
    const double step,
    const double lr,
    const double eps,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const bool amsgrad,
    const int64_t total_element,
    const int64_t global_range) {
  auto& queue = dpcppGetCurrentQueue();

  auto weight_ptr = weight.data_ptr<scalar_t>();
  auto grad_ptr = grad.data_ptr<scalar_t>();
  auto exp_avg_ptr = avg.data_ptr<scalar_t>();
  auto exp_avg_sq_ptr = avg_sq.data_ptr<scalar_t>();
  auto max_exp_avg_sq_ptr = amsgrad ? max_avg_sq.data_ptr<scalar_t>() : nullptr;

  // pre calculate on host side
  auto exp_avg_ele_coefficient = static_cast<float>(1 - beta1);
  auto exp_avg_sq_ele_coefficient = static_cast<float>(1 - beta2);
  auto bias_correction1 =
      static_cast<float>(1.0 - std::pow(static_cast<float>(beta1), step));
  auto bias_correction2 =
      static_cast<float>(1.0 - std::pow(static_cast<float>(beta2), step));
  auto step_size = static_cast<float>(lr / bias_correction1);

  // vec_t is used for vectorization weight and grad
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(sycl::range<1>{global_range}, [=](sycl::item<1> item) {
      auto id = item.get_id(0);

      // cast grad, weight and other memory using vector
      vec_t* weight_vec = reinterpret_cast<vec_t*>(weight_ptr);
      vec_t* grad_vec = reinterpret_cast<vec_t*>(grad_ptr);
      vec_t* exp_avg_vec = reinterpret_cast<vec_t*>(exp_avg_ptr);
      vec_t* exp_avg_sq_vec = reinterpret_cast<vec_t*>(exp_avg_sq_ptr);

      // if amsgrad is false, max_exp_avg_sq_vec is nullptr
      vec_t* max_exp_avg_sq_vec =
          amsgrad ? reinterpret_cast<vec_t*>(max_exp_avg_sq_ptr) : nullptr;

      auto remaining = total_element - id * vec_size;

      // for handling remaining tail
      if (remaining < vec_size) {
        for (auto v_index = 0; v_index < remaining; ++v_index) {
          // kick out tail
          auto linear_idx = id * vec_size + v_index;
          // master weight grad should be fp32 to involve in computation to keep
          // acc.
          auto grad_elm = grad_ptr[linear_idx];
          auto weight_elem = weight_ptr[linear_idx];
          weight_elem = weight_elem - weight_elem * (lr * weight_decay);

          // exp_avg
          auto exp_avg_ele = exp_avg_ptr[linear_idx];
          exp_avg_ele =
              exp_avg_ele * beta1 + grad_elm * exp_avg_ele_coefficient;
          exp_avg_ptr[linear_idx] = static_cast<scalar_t>(exp_avg_ele);

          // exp_avg_sq
          auto exp_avg_sq_ele = exp_avg_sq_ptr[linear_idx];
          exp_avg_sq_ele = exp_avg_sq_ele * beta2 +
              grad_elm * grad_elm * exp_avg_sq_ele_coefficient;
          exp_avg_sq_ptr[linear_idx] = static_cast<scalar_t>(exp_avg_sq_ele);

          // amsgrad
          if (amsgrad) {
            max_exp_avg_sq_ptr[linear_idx] =
                max_exp_avg_sq_ptr[linear_idx] < exp_avg_sq_ele
                ? static_cast<scalar_t>(exp_avg_sq_ele)
                : max_exp_avg_sq_ptr[linear_idx];
            weight_elem = weight_elem -
                step_size * exp_avg_ele /
                    (Numerics<float>::sqrt(
                         max_exp_avg_sq_ptr[linear_idx] / bias_correction2) +
                     eps);
          } else {
            weight_elem = weight_elem -
                step_size * exp_avg_ele /
                    (Numerics<float>::sqrt(exp_avg_sq_ele / bias_correction2) +
                     eps);
          }

          // update real weight
          weight_ptr[linear_idx] = static_cast<scalar_t>(weight_elem);
        }
      } else {
        // vector read
        vec_t weight_value = weight_vec[id];
        vec_t grad_value = grad_vec[id];
        vec_t exp_avg_value = exp_avg_vec[id];
        vec_t exp_avg_sq_value = exp_avg_sq_vec[id];

        // for vector write back
        vec_t temp_weight;
        vec_t temp_exp_avg;
        vec_t temp_exp_avg_sq;

#pragma unroll
        for (auto v_index = 0; v_index < vec_size; ++v_index) {
          auto grad_elm = grad_value[v_index];

          auto weight_elem = weight_value[v_index];
          weight_elem = weight_elem - weight_elem * (lr * weight_decay);

          // exp_avg
          auto exp_avg_ele = exp_avg_value[v_index];
          exp_avg_ele =
              exp_avg_ele * beta1 + grad_elm * exp_avg_ele_coefficient;
          temp_exp_avg[v_index] = static_cast<scalar_t>(exp_avg_ele);

          // exp_avg_sq
          auto exp_avg_sq_ele = exp_avg_sq_value[v_index];
          exp_avg_sq_ele = exp_avg_sq_ele * beta2 +
              grad_elm * grad_elm * exp_avg_sq_ele_coefficient;
          temp_exp_avg_sq[v_index] = static_cast<scalar_t>(exp_avg_sq_ele);

          // amsgrad
          if (amsgrad) {
            auto max_exp_avg_sq_ele = max_exp_avg_sq_vec[id][v_index];
            max_exp_avg_sq_ele = max_exp_avg_sq_ele < exp_avg_sq_ele
                ? static_cast<float>(exp_avg_sq_ele)
                : max_exp_avg_sq_ele;
            max_exp_avg_sq_vec[id][v_index] = max_exp_avg_sq_ele;
            weight_elem = weight_elem -
                step_size * exp_avg_ele /
                    (Numerics<float>::sqrt(
                         max_exp_avg_sq_ele / bias_correction2) +
                     eps);
          } else {
            weight_elem = weight_elem -
                step_size * exp_avg_ele /
                    (Numerics<float>::sqrt(exp_avg_sq_ele / bias_correction2) +
                     eps);
          }

          // update real weight fp32
          temp_weight[v_index] = static_cast<scalar_t>(weight_elem);
        }

        // write back
        // update exp_avg
        exp_avg_vec[id] = temp_exp_avg;

        // update exp_avg_sq
        exp_avg_sq_vec[id] = temp_exp_avg_sq;

        // update weight fp32
        weight_vec[id] = temp_weight;
      }
    });
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

// Here is the AdamW Master Weight kernel
template <typename scalar_t>
static void ComputeAdamWKernelMasterWeight(
    Tensor& master_weight,
    Tensor& avg,
    Tensor& avg_sq,
    Tensor& max_avg_sq,
    Tensor& grad,
    Tensor& weight,
    const bool amsgrad,
    const double step,
    const double beta1,
    const double beta2,
    const double lr,
    const double weight_decay,
    const double eps) {
  auto& queue = dpcppGetCurrentQueue();

  auto vec_size_master_weight =
      at::native::Memory::can_vectorize_up_to_loop<float>(
          getDeviceIdOfCurrentQueue(),
          reinterpret_cast<char*>(master_weight.data_ptr<float>()));

  auto vec_size_weight = at::native::Memory::can_vectorize_up_to_loop<scalar_t>(
      getDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(weight.data_ptr<scalar_t>()));

  auto vec_size = std::min(vec_size_master_weight, vec_size_weight);

  auto total_element = master_weight.numel();

  auto global_range = (total_element % vec_size == 0)
      ? (total_element / vec_size)
      : (total_element / vec_size + 1);

// launch vector kernel for AdamW, code pass according to vector size
#define VEC_ADAMWMW_KERNEL(vec_size)                         \
  {                                                          \
    launch_vec_kernel_AdamWMasterWeight<vec_size, scalar_t>( \
        master_weight,                                       \
        weight,                                              \
        grad,                                                \
        avg,                                                 \
        avg_sq,                                              \
        max_avg_sq,                                          \
        step,                                                \
        lr,                                                  \
        eps,                                                 \
        beta1,                                               \
        beta2,                                               \
        weight_decay,                                        \
        amsgrad,                                             \
        total_element,                                       \
        global_range);                                       \
  }

  switch (vec_size) {
    case 8: {
      VEC_ADAMWMW_KERNEL(8);
      break;
    }
    case 4: {
      VEC_ADAMWMW_KERNEL(4);
      break;
    }
    case 2: {
      VEC_ADAMWMW_KERNEL(2);
      break;
    }
    case 1: {
      VEC_ADAMWMW_KERNEL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for AdamW Master Weight kernel. vec size ",
          vec_size);
  }
#undef VEC_ADAMWMW_KERNEL
}

// no master weight
template <typename scalar_t>
static void ComputeAdamWKernel(
    Tensor& weight,
    Tensor& avg,
    Tensor& avg_sq,
    Tensor& max_avg_sq,
    Tensor& grad,
    const bool amsgrad,
    const double step,
    const double beta1,
    const double beta2,
    const double lr,
    const double weight_decay,
    const double eps) {
  auto& queue = dpcppGetCurrentQueue();

  auto vec_size = at::native::Memory::can_vectorize_up_to_loop<scalar_t>(
      getDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(weight.data_ptr<scalar_t>()));

  auto total_element = weight.numel();

  auto global_range = (total_element % vec_size == 0)
      ? (total_element / vec_size)
      : (total_element / vec_size + 1);

#define VEC_ADAMW_KERNEL(vec_size)               \
  {                                              \
    launch_vec_kernel_AdamW<vec_size, scalar_t>( \
        weight,                                  \
        grad,                                    \
        avg,                                     \
        avg_sq,                                  \
        max_avg_sq,                              \
        step,                                    \
        lr,                                      \
        eps,                                     \
        beta1,                                   \
        beta2,                                   \
        weight_decay,                            \
        amsgrad,                                 \
        total_element,                           \
        global_range);                           \
  }

  switch (vec_size) {
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
} // namespace AtenIpexTypeXPU
} // namespace impl

// fusing adamw kernel and using vector load/store
void fused_ADAMW(
    at::Tensor& param_,
    at::Tensor& exp_avg_,
    at::Tensor& exp_avg_sq_,
    at::Tensor& max_exp_avg_sq_,
    at::Tensor& grad_,
    at::Tensor& param2_,
    const bool amsgrad,
    const double step,
    const double beta1,
    const double beta2,
    const double learning_rate,
    const double weight_decay,
    const double eps) {
  TORCH_CHECK(
      learning_rate >= 0, "Expect learning rate >= 0.0, got ", learning_rate);
  TORCH_CHECK(eps >= 0, "Expect eps >= 0.0, got ", eps);
  TORCH_CHECK(beta1 >= 0 && beta1 < 1, "Expect 0.0 <= beta1 < 1.0, got", beta1);
  TORCH_CHECK(beta2 >= 0 && beta2 < 1, "Expect 0.0 <= beta2 < 1.0, got", beta2);
  TORCH_CHECK(
      weight_decay >= 0, "Expect weight_decay >= 0.0, got ", weight_decay);

  TORCH_CHECK(
      param_.sizes() == grad_.sizes(),
      "Expect param and grad have the same sizes, param sizes: ",
      param_.sizes(),
      "; grad sizes: ",
      grad_.sizes());
  TORCH_CHECK(
      param_.sizes() == exp_avg_.sizes(),
      "Expect param and exp_avg have the same sizes, param sizes: ",
      param_.sizes(),
      "; exp_avg sizes: ",
      exp_avg_.sizes());
  TORCH_CHECK(
      param_.sizes() == exp_avg_sq_.sizes(),
      "Expect param and exp_avg_sq_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; exp_avg_sq sizes: ",
      exp_avg_sq_.sizes());
  if (amsgrad) {
    TORCH_CHECK(
        param_.sizes() == max_exp_avg_sq_.sizes(),
        "Expect param and max_exp_avg_sq_ have the same sizes, param sizes: ",
        param_.sizes(),
        "; max_exp_avg_sq sizes: ",
        max_exp_avg_sq_.sizes());
  }
  TORCH_CHECK(
      param2_.numel() == 0 || param_.sizes() == param2_.sizes(),
      "Expect param and param2_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; param2_ sizes: ",
      param2_.sizes());
  RECORD_FUNCTION(
      "fused_ADAMW",
      std::vector<c10::IValue>(
          {param_, exp_avg_, exp_avg_sq_, max_exp_avg_sq_, grad_, param2_}));

  // support contiguous and channels_last contiguous
  auto memory_format = param_.suggest_memory_format();
  param_ = param_.contiguous(memory_format);
  exp_avg_ = exp_avg_.contiguous(memory_format);
  exp_avg_sq_ = exp_avg_sq_.contiguous(memory_format);
  if (amsgrad) {
    max_exp_avg_sq_ = max_exp_avg_sq_.contiguous(memory_format);
  }
  grad_ = grad_.contiguous(memory_format);

  if (param2_.numel()) {
    // master weight mode
    param2_ = param2_.contiguous(memory_format);
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        param2_.scalar_type(),
        "fused_ADAMW",
        [&] {
          impl::ComputeAdamWKernelMasterWeight<scalar_t>(
              param_,
              exp_avg_,
              exp_avg_sq_,
              max_exp_avg_sq_,
              grad_,
              param2_,
              amsgrad,
              step,
              beta1,
              beta2,
              learning_rate,
              weight_decay,
              eps);
        });
  } else {
    // normal mode, no master weight, all fp32, scalar_t = fp32
    IPEX_DISPATCH_FLOATING_TYPES(param_.scalar_type(), "fused_ADAMW", [&] {
      impl::ComputeAdamWKernel<scalar_t>(
          param_,
          exp_avg_,
          exp_avg_sq_,
          max_exp_avg_sq_,
          grad_,
          amsgrad,
          step,
          beta1,
          beta2,
          learning_rate,
          weight_decay,
          eps);
    });
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
