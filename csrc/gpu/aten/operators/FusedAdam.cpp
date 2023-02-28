#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/record_function.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <torch/library.h>
#include <utils/DPCPP.h>
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"

#include "comm/ATDispatch.h"
#include "comm/Numerics.h"

#include <aten/operators/MemoryAccess.h>
#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

template <int vec_size, typename scalar_t>
void launch_vec_kernel_AdamMasterWeight(
    Tensor& master_weight,
    Tensor& weight,
    Tensor& grad,
    Tensor& avg,
    Tensor& avg_sq,
    Tensor& max_avg_sq,
    const bool amsgrad,
    const bool use_weight_decay,
    const float exp_avg_ele_coefficient,
    const float exp_avg_sq_ele_coefficient,
    const float beta1_value,
    const float beta2_value,
    const float bias_correction1,
    const float bias_correction2,
    const float step_size,
    const float weight_decay,
    const float eps_value,
    const int64_t total_element,
    const int64_t global_range) {
  auto& queue = dpcppGetCurrentQueue();

  auto master_weight_ptr = master_weight.data_ptr<float>();
  auto weight_ptr = weight.data_ptr<scalar_t>();
  auto grad_ptr = grad.data_ptr<scalar_t>();
  auto exp_avg_ptr = avg.data_ptr<float>();
  auto exp_avg_sq_ptr = avg_sq.data_ptr<float>();
  auto max_exp_avg_sq_ptr = amsgrad ? max_avg_sq.data_ptr<float>() : nullptr;

  // vec_t is used for vectorization weight and grad
  using vec_t = at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;
  // vec_mw_t is used for vectorization master weight, exp, exp_sq and
  // max_exp_sq
  using vec_mw_t = at::native::Memory::aligned_vector_loop<float, vec_size>;

  // cast grad, weight and other memory using vector
  vec_t* grad_vec = reinterpret_cast<vec_t*>(grad_ptr);
  vec_t* weight_vec = reinterpret_cast<vec_t*>(weight_ptr);
  vec_mw_t* master_weight_vec = reinterpret_cast<vec_mw_t*>(master_weight_ptr);
  vec_mw_t* exp_avg_vec = reinterpret_cast<vec_mw_t*>(exp_avg_ptr);
  vec_mw_t* exp_avg_sq_vec = reinterpret_cast<vec_mw_t*>(exp_avg_sq_ptr);

  // if amsgrad is false, max_exp_avg_sq_vec is nullptr
  vec_mw_t* max_exp_avg_sq_vec =
      amsgrad ? reinterpret_cast<vec_mw_t*>(max_exp_avg_sq_ptr) : nullptr;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(sycl::range<1>{global_range}, [=](sycl::item<1> item) {
      auto id = item.get_id(0);

      auto remaining = total_element - id * vec_size;

      if (remaining < vec_size) {
        // for the remaining tail elements
        for (auto v_index = 0; v_index < remaining; v_index++) {
          // compute the real index
          auto linear_idx = id * vec_size + v_index;
          // master weight grad should be fp32 to involve in computation to keep
          // acc.
          auto grad_elm = static_cast<float>(grad_ptr[linear_idx]);

          auto master_weight_elm = master_weight_ptr[linear_idx];
          if (use_weight_decay) {
            grad_elm += master_weight_elm * weight_decay;
          }

          // exp_avg
          auto exp_avg_elm = exp_avg_ptr[linear_idx];
          exp_avg_elm =
              exp_avg_elm * beta1_value + grad_elm * exp_avg_ele_coefficient;
          exp_avg_ptr[linear_idx] = exp_avg_elm;

          // exp_avg_sq
          auto exp_avg_sq_elm = exp_avg_sq_ptr[linear_idx];
          exp_avg_sq_elm = exp_avg_sq_elm * beta2_value +
              grad_elm * grad_elm * exp_avg_sq_ele_coefficient;
          exp_avg_sq_ptr[linear_idx] = exp_avg_sq_elm;

          // amsgrad
          if (amsgrad) {
            auto max_exp_avg_sq_elm = max_exp_avg_sq_ptr[linear_idx];
            max_exp_avg_sq_elm = max_exp_avg_sq_elm < exp_avg_sq_elm
                ? exp_avg_sq_elm
                : max_exp_avg_sq_elm;
            max_exp_avg_sq_ptr[linear_idx] = max_exp_avg_sq_elm;
            master_weight_elm -= step_size * exp_avg_elm /
                (Numerics<float>::sqrt(max_exp_avg_sq_elm / bias_correction2) +
                 eps_value);
          } else {
            master_weight_elm -= step_size * exp_avg_elm /
                (Numerics<float>::sqrt(exp_avg_sq_elm / bias_correction2) +
                 eps_value);
          }

          // update master weight fp32
          master_weight_ptr[linear_idx] = static_cast<float>(master_weight_elm);
          // update real weight bf16/fp16
          weight_ptr[linear_idx] = static_cast<scalar_t>(master_weight_elm);
        }
      } else {
        // can use vectorization
        vec_mw_t exp_avg_value = exp_avg_vec[id];
        vec_mw_t exp_avg_sq_value = exp_avg_sq_vec[id];
        vec_mw_t master_weight_value = master_weight_vec[id];

        // for vector write back
        vec_t temp_weight;
        vec_mw_t temp_master_weight;
        vec_mw_t temp_exp_avg;
        vec_mw_t temp_exp_avg_sq;

#pragma unroll(vec_size)
        for (auto v_index = 0; v_index < vec_size; v_index++) {
          auto grad_elm = static_cast<float>(grad_vec[id][v_index]);

          auto master_weight_elm = master_weight_value[v_index];
          if (use_weight_decay) {
            grad_elm += master_weight_elm * weight_decay;
          }
          // exp_avg
          auto exp_avg_elm = exp_avg_value[v_index];
          exp_avg_elm =
              exp_avg_elm * beta1_value + exp_avg_ele_coefficient * grad_elm;
          temp_exp_avg[v_index] = exp_avg_elm;

          // exp_avg_sq
          auto exp_avg_sq_elm = exp_avg_sq_value[v_index];
          exp_avg_sq_elm = exp_avg_sq_elm * beta2_value +
              grad_elm * grad_elm * exp_avg_sq_ele_coefficient;
          temp_exp_avg_sq[v_index] = exp_avg_sq_elm;

          // amsgrad
          if (amsgrad) {
            auto max_exp_avg_sq_elm = max_exp_avg_sq_vec[id][v_index];
            max_exp_avg_sq_elm = max_exp_avg_sq_elm < exp_avg_sq_elm
                ? exp_avg_sq_elm
                : max_exp_avg_sq_elm;
            max_exp_avg_sq_vec[id][v_index] = max_exp_avg_sq_elm;
            master_weight_elm -= step_size * exp_avg_elm /
                (Numerics<float>::sqrt(max_exp_avg_sq_elm / bias_correction2) +
                 eps_value);
          } else {
            master_weight_elm -= step_size * exp_avg_elm /
                (Numerics<float>::sqrt(exp_avg_sq_elm / bias_correction2) +
                 eps_value);
          }

          // update master weight with fp32
          temp_master_weight[v_index] = static_cast<float>(master_weight_elm);
          // update real weight with bf16/fp16
          temp_weight[v_index] = static_cast<scalar_t>(master_weight_elm);
        }

        // write_back
        // update exp_avg
        exp_avg_vec[id] = temp_exp_avg;
        // update exp_avg_sq
        exp_avg_sq_vec[id] = temp_exp_avg_sq;
        // update master weight
        master_weight_vec[id] = temp_master_weight;
        // update real weight
        weight_vec[id] = temp_weight;
      }
    });
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

template <int vec_size>
void launch_vec_kernel_Adam(
    Tensor& weight,
    Tensor& grad,
    Tensor& avg,
    Tensor& avg_sq,
    Tensor& max_avg_sq,
    const bool amsgrad,
    const bool use_weight_decay,
    const int step,
    const float exp_avg_ele_coefficient,
    const float exp_avg_sq_ele_coefficient,
    const float beta1_value,
    const float beta2_value,
    const float bias_correction1,
    const float bias_correction2,
    const float step_size,
    const float weight_decay,
    const float eps_value,
    const int64_t total_element,
    const int64_t global_range) {
  auto& queue = dpcppGetCurrentQueue();
  auto weight_ptr = weight.data_ptr<float>();
  auto grad_ptr = grad.data_ptr<float>();
  auto exp_avg_ptr = avg.data_ptr<float>();
  auto exp_avg_sq_ptr = avg_sq.data_ptr<float>();
  auto max_exp_avg_sq_ptr = amsgrad ? max_avg_sq.data_ptr<float>() : nullptr;

  using vec_t = at::native::Memory::aligned_vector_loop<float, vec_size>;

  // cast grad, weight and other memory using vector
  vec_t* weight_vec = reinterpret_cast<vec_t*>(weight_ptr);
  vec_t* grad_vec = reinterpret_cast<vec_t*>(grad_ptr);
  vec_t* exp_avg_vec = reinterpret_cast<vec_t*>(exp_avg_ptr);
  vec_t* exp_avg_sq_vec = reinterpret_cast<vec_t*>(exp_avg_sq_ptr);
  vec_t* max_exp_avg_sq_vec =
      amsgrad ? reinterpret_cast<vec_t*>(max_exp_avg_sq_ptr) : nullptr;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(sycl::range<1>{global_range}, [=](sycl::item<1> item) {
      auto id = item.get_id(0);
      auto remaining = total_element - id * vec_size;

      // for handling remaining tail
      if (remaining < vec_size) {
        for (auto v_index = 0; v_index < remaining; v_index++) {
          // compute element index
          auto linear_idx = id * vec_size + v_index;
          auto grad_elm = grad_ptr[linear_idx];
          auto weight_elm = weight_ptr[linear_idx];

          if (use_weight_decay) {
            grad_elm += weight_elm * weight_decay;
          }

          // exp_avg
          auto exp_avg_elm = exp_avg_ptr[linear_idx];
          exp_avg_elm =
              exp_avg_elm * beta1_value + grad_elm * exp_avg_ele_coefficient;
          exp_avg_ptr[linear_idx] = exp_avg_elm;

          // exp_avg_sq
          auto exp_avg_sq_elm = exp_avg_sq_ptr[linear_idx];
          exp_avg_sq_elm = exp_avg_sq_elm * beta2_value +
              exp_avg_sq_ele_coefficient * grad_elm * grad_elm;
          exp_avg_sq_ptr[linear_idx] = exp_avg_sq_elm;

          // amsgrad
          if (amsgrad) {
            auto max_exp_avg_sq_elm = max_exp_avg_sq_ptr[linear_idx];
            max_exp_avg_sq_elm = max_exp_avg_sq_elm < exp_avg_sq_elm
                ? exp_avg_sq_elm
                : max_exp_avg_sq_elm;
            max_exp_avg_sq_ptr[linear_idx] = max_exp_avg_sq_elm;
            weight_elm = weight_elm -
                step_size * exp_avg_elm /
                    (Numerics<float>::sqrt(
                         max_exp_avg_sq_elm / bias_correction2) +
                     eps_value);
          } else {
            weight_elm = weight_elm -
                step_size * exp_avg_elm /
                    (Numerics<float>::sqrt(exp_avg_sq_elm / bias_correction2) +
                     eps_value);
          }

          weight_ptr[linear_idx] = static_cast<float>(weight_elm);
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
        for (auto v_index = 0; v_index < vec_size; v_index++) {
          auto grad_elm = grad_value[v_index];
          auto weight_elm = weight_value[v_index];

          if (use_weight_decay) {
            grad_elm += weight_elm * weight_decay;
          }

          // exp_avg
          auto exp_avg_elm = exp_avg_value[v_index];
          exp_avg_elm =
              beta1_value * exp_avg_elm + grad_elm * exp_avg_ele_coefficient;
          temp_exp_avg[v_index] = exp_avg_elm;

          // exp_avg_sq
          auto exp_avg_sq_elm = exp_avg_sq_value[v_index];
          exp_avg_sq_elm = beta2_value * exp_avg_sq_elm +
              exp_avg_sq_ele_coefficient * grad_elm * grad_elm;
          temp_exp_avg_sq[v_index] = exp_avg_sq_elm;

          // amsgrad
          if (amsgrad) {
            auto max_exp_avg_sq_elm = max_exp_avg_sq_vec[id][v_index];
            max_exp_avg_sq_elm =
                Numerics<float>::max(exp_avg_sq_elm, max_exp_avg_sq_elm);
            weight_elm -= step_size * exp_avg_elm /
                (Numerics<float>::sqrt(max_exp_avg_sq_elm / bias_correction2) +
                 eps_value);
            max_exp_avg_sq_vec[id][v_index] = max_exp_avg_sq_elm;
          } else {
            weight_elm = weight_elm -
                (step_size * exp_avg_elm /
                 (Numerics<float>::sqrt(exp_avg_sq_elm / bias_correction2) +
                  eps_value));
          }
          temp_weight[v_index] = static_cast<float>(weight_elm);
        }

        // data write back
        weight_vec[id] = temp_weight;
        exp_avg_vec[id] = temp_exp_avg;
        exp_avg_sq_vec[id] = temp_exp_avg_sq;
      }
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

static void ComputeAdamKernel(
    Tensor& weight,
    Tensor& avg,
    Tensor& avg_sq,
    Tensor& max_avg_sq,
    Tensor& grad,
    const bool amsgrad,
    const bool use_weight_decay,
    const int step,
    const float exp_avg_ele_coefficient,
    const float exp_avg_sq_ele_coefficient,
    const float beta1_value,
    const float beta2_value,
    const float bias_correlation1,
    const float bias_correlation2,
    const float step_size,
    const float weight_decay,
    const float eps_value) {
  auto& queue = dpcppGetCurrentQueue();

  auto vec_size_weight = at::native::Memory::can_vectorize_up_to_loop<float>(
      getDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(weight.data_ptr<float>()));

  auto vec_size_grad = at::native::Memory::can_vectorize_up_to_loop<float>(
      getDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(grad.data_ptr<float>()));

  auto vec_size_avg = at::native::Memory::can_vectorize_up_to_loop<float>(
      getDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(avg.data_ptr<float>()));

  auto vec_size_avg_sq = at::native::Memory::can_vectorize_up_to_loop<float>(
      getDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(avg_sq.data_ptr<float>()));

  auto vec_size_avg_max_sq = vec_size_avg_sq;
  if (amsgrad) {
    vec_size_avg_max_sq = at::native::Memory::can_vectorize_up_to_loop<float>(
        getDeviceIdOfCurrentQueue(),
        reinterpret_cast<char*>(max_avg_sq.data_ptr<float>()));
  }
  // for non-dense
  auto vec_size = vec_size_weight;
  if (!weight.is_non_overlapping_and_dense() ||
      !grad.is_non_overlapping_and_dense() ||
      !avg.is_non_overlapping_and_dense() ||
      !avg_sq.is_non_overlapping_and_dense() ||
      (amsgrad && !max_avg_sq.is_non_overlapping_and_dense())) {
    vec_size = 1;
  } else {
    vec_size = std::min(
        {vec_size_weight,
         vec_size_grad,
         vec_size_avg,
         vec_size_avg_sq,
         vec_size_avg_max_sq});
  }

  auto total_element = weight.numel();

  auto global_range = (total_element % vec_size == 0)
      ? (total_element / vec_size)
      : (total_element / vec_size + 1);

#define VEC_ADAM_KERNEL(vec_size)     \
  {                                   \
    launch_vec_kernel_Adam<vec_size>( \
        weight,                       \
        grad,                         \
        avg,                          \
        avg_sq,                       \
        max_avg_sq,                   \
        amsgrad,                      \
        use_weight_decay,             \
        step,                         \
        exp_avg_ele_coefficient,      \
        exp_avg_sq_ele_coefficient,   \
        beta1_value,                  \
        beta2_value,                  \
        bias_correlation1,            \
        bias_correlation2,            \
        step_size,                    \
        weight_decay,                 \
        eps_value,                    \
        total_element,                \
        global_range);                \
  }

  switch (vec_size) {
    case 4:
      VEC_ADAM_KERNEL(4);
      break;
    case 2:
      VEC_ADAM_KERNEL(2);
      break;
    case 1:
      VEC_ADAM_KERNEL(1);
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for Adam kernel. vec size ",
          vec_size);
  }
#undef VEC_ADAM_KERNEL
}
template <typename scalar_t>
static void ComputeAdamKernelMasterWeight(
    Tensor& master_weight,
    Tensor& avg,
    Tensor& avg_sq,
    Tensor& max_avg_sq,
    Tensor& grad,
    Tensor& weight,
    const bool amsgrad,
    const bool use_weight_decay,
    const float exp_avg_ele_coefficient,
    const float exp_avg_sq_ele_coefficient,
    const float beta1_value,
    const float beta2_value,
    const float bias_correction1,
    const float bias_correction2,
    const float step_size,
    const float weight_decay,
    const float eps_value) {
  auto& queue = dpcppGetCurrentQueue();

  auto vec_size_master_weight =
      at::native::Memory::can_vectorize_up_to_loop<float>(
          getDeviceIdOfCurrentQueue(),
          reinterpret_cast<char*>(master_weight.data_ptr<float>()));

  auto vec_size_weight = at::native::Memory::can_vectorize_up_to_loop<scalar_t>(
      getDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(weight.data_ptr<scalar_t>()));

  auto vec_size_grad = at::native::Memory::can_vectorize_up_to_loop<scalar_t>(
      getDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(grad.data_ptr<scalar_t>()));

  auto vec_size_avg = at::native::Memory::can_vectorize_up_to_loop<float>(
      getDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(avg.data_ptr<float>()));

  auto vec_size_avg_sq = at::native::Memory::can_vectorize_up_to_loop<float>(
      getDeviceIdOfCurrentQueue(),
      reinterpret_cast<char*>(avg_sq.data_ptr<float>()));

  auto vec_size_max_avg_sq = vec_size_avg_sq;
  if (amsgrad) {
    vec_size_max_avg_sq = at::native::Memory::can_vectorize_up_to_loop<float>(
        getDeviceIdOfCurrentQueue(),
        reinterpret_cast<char*>(max_avg_sq.data_ptr<float>()));
  }

  auto vec_size = vec_size_master_weight;
  if (!master_weight.is_non_overlapping_and_dense() ||
      !weight.is_non_overlapping_and_dense() ||
      !grad.is_non_overlapping_and_dense() ||
      !avg.is_non_overlapping_and_dense() ||
      !avg_sq.is_non_overlapping_and_dense() ||
      (amsgrad && !max_avg_sq.is_non_overlapping_and_dense())) {
    vec_size = 1;
  } else {
    vec_size = std::min(
        {vec_size_master_weight,
         vec_size_weight,
         vec_size_grad,
         vec_size_avg,
         vec_size_avg_sq,
         vec_size_max_avg_sq});
  }

  auto total_element = master_weight.numel();
  auto global_range = (total_element % vec_size == 0)
      ? (total_element / vec_size)
      : (total_element / vec_size + 1);

#define VEC_ADAMMW_KERNEL(vec_size)                         \
  {                                                         \
    launch_vec_kernel_AdamMasterWeight<vec_size, scalar_t>( \
        master_weight,                                      \
        weight,                                             \
        grad,                                               \
        avg,                                                \
        avg_sq,                                             \
        max_avg_sq,                                         \
        amsgrad,                                            \
        use_weight_decay,                                   \
        exp_avg_ele_coefficient,                            \
        exp_avg_sq_ele_coefficient,                         \
        beta1_value,                                        \
        beta2_value,                                        \
        bias_correction1,                                   \
        bias_correction2,                                   \
        step_size,                                          \
        weight_decay,                                       \
        eps_value,                                          \
        total_element,                                      \
        global_range);                                      \
  }
  switch (vec_size) {
    case 8: {
      VEC_ADAMMW_KERNEL(8);
      break;
    }
    case 4: {
      VEC_ADAMMW_KERNEL(4);
      break;
    }
    case 2: {
      VEC_ADAMMW_KERNEL(2);
      break;
    }
    case 1: {
      VEC_ADAMMW_KERNEL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for Adam Master Weight kernel. vec size ",
          vec_size);
  }
#undef VEC_ADAMMW_KERNEL
}

} // namespace impl
void adam_fused_step(
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
  // check whether enable param foreach
  TORCH_CHECK(
      learning_rate > 0, "Expect learning rate >= 0.0, got ", learning_rate);
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
      "adam_fused_step",
      std::vector<c10::IValue>(
          {param_, exp_avg_, exp_avg_sq_, max_exp_avg_sq_, grad_, param2_}));

  const OptionalDeviceGuard device_guard(device_of(param_));

  // after inference, the model weight in the next training epoch maybe cached
  // in block layout, so to plain now if needed
  param_ = to_plain_if_needed(param_);
  grad_ = to_plain_if_needed(grad_);

  // support contiguous and channels_last contiguous
  auto memory_format = param_.suggest_memory_format();
  param_ = param_.contiguous(memory_format);
  grad_ = grad_.contiguous(memory_format);
  exp_avg_ = exp_avg_.contiguous(memory_format);
  exp_avg_sq_ = exp_avg_sq_.contiguous(memory_format);
  if (amsgrad) {
    max_exp_avg_sq_ = max_exp_avg_sq_.contiguous(memory_format);
  }

  // pre calculate scalar on host side
  bool use_weight_decay = false;
  if (weight_decay != 0) {
    use_weight_decay = true;
  }
  const auto beta1_value = static_cast<float>(beta1);
  const auto beta2_value = static_cast<float>(beta2);
  const auto exp_avg_ele_coefficient = static_cast<float>(1.0 - beta1_value);
  const auto exp_avg_sq_ele_coefficient = static_cast<float>(1.0 - beta2_value);
  const auto bias_correction1 =
      static_cast<float>(1.0 - std::pow(beta1_value, step));
  const auto bias_correction2 =
      static_cast<float>(1.0 - std::pow(beta2_value, step));
  const auto step_size = static_cast<float>(learning_rate / bias_correction1);
  const float weight_decay_value = static_cast<float>(weight_decay);
  const auto eps_value = static_cast<float>(eps);

  if (param2_.numel() != 0) {
    // should use master weight, param2_ is fp32
    param2_ = to_plain_if_needed_(param2_);
    param2_ = param2_.contiguous(memory_format);
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        param2_.scalar_type(),
        "adam_fused_step",
        [&] {
          impl::ComputeAdamKernelMasterWeight<scalar_t>(
              param_,
              exp_avg_,
              exp_avg_sq_,
              max_exp_avg_sq_,
              grad_,
              param2_,
              amsgrad,
              use_weight_decay,
              exp_avg_ele_coefficient,
              exp_avg_sq_ele_coefficient,
              beta1_value,
              beta2_value,
              bias_correction1,
              bias_correction2,
              step_size,
              weight_decay_value,
              eps_value);
        });
  } else {
    // normal mode, param_ is fp32
    impl::ComputeAdamKernel(
        param_,
        exp_avg_,
        exp_avg_sq_,
        max_exp_avg_sq_,
        grad_,
        amsgrad,
        use_weight_decay,
        step,
        exp_avg_ele_coefficient,
        exp_avg_sq_ele_coefficient,
        beta1_value,
        beta2_value,
        bias_correction1,
        bias_correction2,
        step_size,
        weight_decay_value,
        eps_value);
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "adam_fused_step",
      at::AtenIpexTypeXPU::adam_fused_step,
      c10::DispatchKey::XPU);
}
} // namespace
