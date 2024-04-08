#pragma once
#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/record_function.h>
#include <torch/library.h>
#include <utils/DPCPP.h>

#include <ATen/native/ForeachUtils.h>
#include <aten/operators/MemoryAccess.h>
#include "ATen/OpMathType.h"
#include "ForeachFunctors.h"
#include "Loops.h"
#include "MultiTensorApply.h"
#include "comm/Numerics.h"

namespace at {
namespace AtenIpexTypeXPU {

enum class ADAM_MODE : uint8_t { ORIGINAL = 0, ADAMW = 1 };

// index in TensorList for params
constexpr uint8_t kParamIdx = 0;
constexpr uint8_t kGradIdx = 1;
constexpr uint8_t kExpAvgIdx = 2;
constexpr uint8_t kExpAvgSqIdx = 3;
constexpr uint8_t kMaxExpAvgSqIdx = 4;

template <
    typename scalar_type,
    typename opmath_t,
    int depth = 4,
    ADAM_MODE mode = ADAM_MODE::ORIGINAL,
    bool amsgrad>
inline void adam_math(
    scalar_type r_args[depth][kILP],
    const float* step_count,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    const bool maximize,
    const opmath_t bias_correction1,
    const opmath_t step_size,
    const double lr_weight_decay,
    const opmath_t bias_correction2,
    const opmath_t bias_correction2_sqrt) {
  bool use_weight_decay = weight_decay != 0;

#pragma unroll
  for (int ii = 0; ii < kILP; ii++) {
    // Load values.
    opmath_t param = static_cast<opmath_t>(r_args[kParamIdx][ii]);
    opmath_t grad = static_cast<opmath_t>(r_args[kGradIdx][ii]);
    const opmath_t grad_to_store = grad;

    if (maximize) {
      grad = -grad;
    }

    opmath_t exp_avg = static_cast<opmath_t>(r_args[kExpAvgIdx][ii]);
    opmath_t exp_avg_sq = static_cast<opmath_t>(r_args[kExpAvgSqIdx][ii]);
    opmath_t max_exp_avg_sq;

    if constexpr (amsgrad) {
      max_exp_avg_sq = static_cast<opmath_t>(r_args[kMaxExpAvgSqIdx][ii]);
    }

    // Update param, grad, 1st and 2nd order momentum.
    if (use_weight_decay) {
      if constexpr (mode == ADAM_MODE::ORIGINAL)
        grad += param * weight_decay;
      else // ADAM_MODE::ADAMW:
        param -= (lr_weight_decay * param);
    }

    // todo(crcrpar): use lerp
    exp_avg = beta1 * exp_avg + (1.0f - beta1) * grad;
    exp_avg_sq = beta2 * exp_avg_sq + (1.0f - beta2) * grad * grad;
    opmath_t denom;

    if constexpr (amsgrad) {
      max_exp_avg_sq = std::max(max_exp_avg_sq, exp_avg_sq);
      denom = (std::sqrt(max_exp_avg_sq) / bias_correction2_sqrt) + eps;
    } else {
      denom = (std::sqrt(exp_avg_sq) / bias_correction2_sqrt) + eps;
    }
    param -= step_size * exp_avg / denom;

    // Store results.
    r_args[kParamIdx][ii] = param;
    r_args[kExpAvgIdx][ii] = exp_avg;
    r_args[kExpAvgSqIdx][ii] = exp_avg_sq;
    if constexpr (amsgrad) {
      r_args[kMaxExpAvgSqIdx][ii] = max_exp_avg_sq;
    }
  }
}

template <
    typename scalar_type,
    int depth = 4,
    ADAM_MODE mode = ADAM_MODE::ORIGINAL,
    bool amsgrad = false>
struct FusedAdamMathFunctor {
  static_assert(
      depth == 4 || depth == 5,
      "depth of 4 for Adam, depth of 5 for Adam with AMSGrad.");
  using opmath_t = at::opmath_type<scalar_type>;
  template <typename TLA, typename TLW>
  void operator()(
      const int chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item,
      const float* lr_ptr,
      const double lr,
      const double beta1,
      const double beta2,
      const double weight_decay,
      const double eps,
      const bool maximize) const {
    auto group_id = item.get_group(0);
    auto item_id = item.get_local_id(0);
    auto local_range = item.get_local_range(0);

    int tensor_loc = tlWGMeta[group_id].wg_to_tensor;
    int chunk_idx = tlWGMeta[group_id].wg_to_chunk;
    int n = tlAddress[tensor_loc].numel_to_tensor;

    float* step_count =
        reinterpret_cast<float*>(tlAddress[tensor_loc].state_steps_addresses);

    scalar_type* args[depth];
    bool all_aligned =
        init_args<depth>(args, tlAddress, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    scalar_type r_args[depth][kILP];
    double lr_value = lr_ptr ? *lr_ptr : lr;

    const opmath_t bias_correction1 = static_cast<opmath_t>(
        1.0f - Numerics<opmath_t>::pow(beta1, *step_count));
    const opmath_t bias_correction2 = static_cast<opmath_t>(
        1.0f - Numerics<opmath_t>::pow(beta2, *step_count));
    const opmath_t step_size =
        static_cast<opmath_t>(lr_value / bias_correction1);
    const opmath_t bias_correction2_sqrt =
        Numerics<opmath_t>::sqrt(bias_correction2);
    const double lr_weight_decay = lr_value * weight_decay;

    if ((n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned) {
      for (int i_start = item_id;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += local_range) {
#pragma unroll
        for (int i = 0; i < depth; i++) {
          load_store(r_args[i], args[i], 0, i_start);
        }
        adam_math<scalar_type, opmath_t, depth, mode, amsgrad>(
            r_args,
            step_count,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            bias_correction1,
            step_size,
            lr_weight_decay,
            bias_correction2,
            bias_correction2_sqrt);

#pragma unroll
        for (int i = 0; i < depth; i++) {
          if (i != kGradIdx) {
            load_store(args[i], r_args[i], i_start, 0);
          }
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size;
           i_start += local_range * kILP) {
        load_args<depth>(
            r_args, args, i_start, chunk_size, n, item_id, local_range);
        adam_math<scalar_type, opmath_t, depth, mode, amsgrad>(
            r_args,
            step_count,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            bias_correction1,
            step_size,
            lr_weight_decay,
            bias_correction2,
            bias_correction2_sqrt);
#pragma unroll
        for (int i = 0; i < depth; i++) {
          if (i != kGradIdx) {
            store_args(
                args[i],
                r_args[i],
                i_start,
                chunk_size,
                n,
                item_id,
                local_range);
          }
        }
      }
    }
  }
};

} // namespace AtenIpexTypeXPU
} // namespace at
