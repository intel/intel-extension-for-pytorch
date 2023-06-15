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

#include <aten/operators/MemoryAccess.h>
#include "Loops.h"
#include "LoopsTemplates.h"
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

template <typename scalar_t>
static void ComputeAdamKernel(
    Tensor& weight,
    Tensor& avg,
    Tensor& avg_sq,
    Tensor& max_avg_sq,
    Tensor& grad,
    const bool amsgrad,
    const bool use_weight_decay,
    const float exp_avg_ele_coefficient,
    const float exp_avg_sq_ele_coefficient,
    const float beta1_value,
    const float beta2_value,
    const float bias_correlation1,
    const float bias_correlation2,
    const float step_size,
    const float weight_decay,
    const float eps_value) {
  if (amsgrad) {
    at::TensorIterator iter = TensorIteratorConfig()
                                  .add_output(avg)
                                  .add_output(avg_sq)
                                  .add_output(max_avg_sq)
                                  .add_output(weight)
                                  .add_input(weight)
                                  .add_input(grad)
                                  .add_input(avg)
                                  .add_input(avg_sq)
                                  .add_input(max_avg_sq)
                                  .build();
    dpcpp_kernel_multiple_outputs_for_tensor_iter(
        iter,
        [=](scalar_t weight_elem,
            scalar_t grad_elem,
            scalar_t avg_elem,
            scalar_t avg_sq_elem,
            scalar_t max_avg_sq_elem)
            -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
          if (use_weight_decay) {
            grad_elem += weight_elem * weight_decay;
          }
          avg_elem =
              avg_elem * beta1_value + grad_elem * exp_avg_ele_coefficient;

          avg_sq_elem = avg_sq_elem * beta2_value +
              exp_avg_sq_ele_coefficient * grad_elem * grad_elem;

          max_avg_sq_elem =
              max_avg_sq_elem < avg_sq_elem ? avg_sq_elem : max_avg_sq_elem;

          weight_elem = weight_elem -
              step_size * avg_elem /
                  (Numerics<float>::sqrt(max_avg_sq_elem / bias_correlation2) +
                   eps_value);
          return std::tuple<scalar_t, scalar_t, scalar_t, scalar_t>(
              avg_elem, avg_sq_elem, max_avg_sq_elem, weight_elem);
        });
  } else {
    at::TensorIterator iter = TensorIteratorConfig()
                                  .add_output(avg)
                                  .add_output(avg_sq)
                                  .add_output(weight)
                                  .add_input(weight)
                                  .add_input(grad)
                                  .add_input(avg)
                                  .add_input(avg_sq)
                                  .build();
    dpcpp_kernel_multiple_outputs_for_tensor_iter(
        iter,
        [=](scalar_t weight_elem,
            scalar_t grad_elem,
            scalar_t avg_elem,
            scalar_t avg_sq_elem) -> std::tuple<scalar_t, scalar_t, scalar_t> {
          if (use_weight_decay) {
            grad_elem += weight_elem * weight_decay;
          }
          avg_elem =
              avg_elem * beta1_value + grad_elem * exp_avg_ele_coefficient;

          avg_sq_elem = avg_sq_elem * beta2_value +
              exp_avg_sq_ele_coefficient * grad_elem * grad_elem;

          weight_elem = weight_elem -
              step_size * avg_elem /
                  (Numerics<float>::sqrt(avg_sq_elem / bias_correlation2) +
                   eps_value);

          return std::tuple<scalar_t, scalar_t, scalar_t>(
              avg_elem, avg_sq_elem, weight_elem);
        });
  }
}

// scalar_t is for fp16 or bf16, master weight is fp32
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
  if (amsgrad) {
    at::TensorIterator iter = TensorIteratorConfig()
                                  .add_output(avg)
                                  .add_output(avg_sq)
                                  .add_output(max_avg_sq)
                                  .add_output(master_weight)
                                  .add_output(weight)
                                  .add_input(grad)
                                  .add_input(avg)
                                  .add_input(avg_sq)
                                  .add_input(max_avg_sq)
                                  .add_input(master_weight)
                                  .check_all_same_dtype(false)
                                  .build();

    dpcpp_kernel_multiple_outputs_for_tensor_iter(
        iter,
        [=](scalar_t grad_elem,
            float avg_elem,
            float avg_sq_elem,
            float max_avg_sq_elem,
            float master_weight_elem)
            -> std::tuple<float, float, float, float, scalar_t> {
          auto grad_float_elem = static_cast<float>(grad_elem);
          if (use_weight_decay) {
            grad_float_elem += master_weight_elem * weight_decay;
          }
          avg_elem = avg_elem * beta1_value +
              grad_float_elem * exp_avg_ele_coefficient;

          avg_sq_elem = avg_sq_elem * beta2_value +
              exp_avg_sq_ele_coefficient * grad_float_elem * grad_float_elem;

          // amsgrad
          max_avg_sq_elem =
              max_avg_sq_elem < avg_sq_elem ? avg_sq_elem : max_avg_sq_elem;

          master_weight_elem = master_weight_elem -
              step_size * avg_elem /
                  (Numerics<float>::sqrt(max_avg_sq_elem / bias_correction2) +
                   eps_value);

          return std::tuple<float, float, float, float, scalar_t>(
              avg_elem,
              avg_sq_elem,
              max_avg_sq_elem,
              master_weight_elem,
              static_cast<scalar_t>(master_weight_elem));
        });
  } else {
    at::TensorIterator iter = TensorIteratorConfig()
                                  .add_output(avg)
                                  .add_output(avg_sq)
                                  .add_output(master_weight)
                                  .add_output(weight)
                                  .add_input(grad)
                                  .add_input(avg)
                                  .add_input(avg_sq)
                                  .add_input(master_weight)
                                  .check_all_same_dtype(false)
                                  .build();

    dpcpp_kernel_multiple_outputs_for_tensor_iter(
        iter,
        [=](scalar_t grad_elem,
            float avg_elem,
            float avg_sq_elem,
            float master_weight_elem)
            -> std::tuple<float, float, float, scalar_t> {
          auto grad_float_elem = static_cast<float>(grad_elem);
          if (use_weight_decay) {
            grad_float_elem += master_weight_elem * weight_decay;
          }
          avg_elem = avg_elem * beta1_value +
              grad_float_elem * exp_avg_ele_coefficient;

          avg_sq_elem = avg_sq_elem * beta2_value +
              exp_avg_sq_ele_coefficient * grad_float_elem * grad_float_elem;

          master_weight_elem = master_weight_elem -
              step_size * avg_elem /
                  (Numerics<float>::sqrt(avg_sq_elem / bias_correction2) +
                   eps_value);

          return std::tuple<float, float, float, scalar_t>(
              avg_elem,
              avg_sq_elem,
              master_weight_elem,
              static_cast<scalar_t>(master_weight_elem));
        });
  }
}
} // namespace impl

void adam_fused_step(
    const at::Tensor& param,
    const at::Tensor& exp_avg,
    const at::Tensor& exp_avg_sq,
    const at::Tensor& max_exp_avg_sq,
    const at::Tensor& grad,
    const at::Tensor& param2,
    bool amsgrad,
    double step,
    double beta1,
    double beta2,
    double learning_rate,
    double weight_decay,
    double eps) {
  // need to keep align with cpu semantic, conver tensor to non-const here
  auto param_ = const_cast<at::Tensor&>(param);
  auto exp_avg_ = const_cast<at::Tensor&>(exp_avg);
  auto exp_avg_sq_ = const_cast<at::Tensor&>(exp_avg_sq);
  auto max_exp_avg_sq_ = const_cast<at::Tensor&>(max_exp_avg_sq);
  auto grad_ = const_cast<at::Tensor&>(grad);
  auto param2_ = const_cast<at::Tensor&>(param2);
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
    // normal mode, param_ is fp32 or fp64
    IPEX_DISPATCH_FLOATING_TYPES(param_.scalar_type(), "adam_fused_step", [&] {
      impl::ComputeAdamKernel<scalar_t>(
          param_,
          exp_avg_,
          exp_avg_sq_,
          max_exp_avg_sq_,
          grad_,
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
