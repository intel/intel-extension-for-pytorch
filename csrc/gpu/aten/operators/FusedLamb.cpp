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
#include <algorithm>
#include "Loops.h"
#include "LoopsTemplates.h"
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"

#include <aten/operators/MemoryAccess.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {

void launch_vec_kernel_Lamb(
    Tensor& weight,
    Tensor& grad,
    Tensor& avg,
    Tensor& avg_sq,
    const int step,
    const double beta1,
    const double beta2,
    const double bias_correction1,
    const double bias_correction2,
    const double learning_rate,
    const double weight_decay,
    const double eps) {
  auto& queue = dpcppGetCurrentQueue();

  Tensor param_norm_tensor = at::zeros_like(weight);
  Tensor rtw_norm_tensor = at::zeros_like(weight);

  // use grad to restore adam_step_elem
  at::TensorIterator iter = TensorIteratorConfig()
                                .add_output(param_norm_tensor)
                                .add_output(rtw_norm_tensor)
                                .add_output(grad)
                                .add_output(avg)
                                .add_output(avg_sq)
                                .add_input(weight)
                                .add_input(grad)
                                .add_input(avg)
                                .add_input(avg_sq)
                                .check_all_same_dtype(false)
                                .build();

  dpcpp_kernel_multiple_outputs_for_tensor_iter(
      iter,
      [=](float weight_elem,
          float grad_elem,
          float exp_avg_elem,
          float exp_avg_sq_elem)
          -> std::tuple<float, float, float, float, float> {
        exp_avg_elem = exp_avg_elem * beta1 + (1 - beta1) * grad_elem;
        exp_avg_sq_elem =
            exp_avg_sq_elem * beta2 + grad_elem * grad_elem * (1 - beta2);
        auto adam_step_elem = exp_avg_elem / bias_correction1 /
            (Numerics<float>::sqrt(exp_avg_sq_elem / bias_correction2) + eps);
        adam_step_elem = adam_step_elem + weight_elem * weight_decay;

        return std::tuple<float, float, float, float, float>{
            weight_elem * weight_elem,
            adam_step_elem * adam_step_elem,
            adam_step_elem,
            exp_avg_elem,
            exp_avg_sq_elem};
      });

  auto param_norm_sum = at::sum(param_norm_tensor).item().to<float>();
  auto rtw_norm_sum = at::sum(rtw_norm_tensor).item().to<float>();
  float true_ratio = std::sqrt(param_norm_sum) / std::sqrt(rtw_norm_sum);

  at::TensorIterator update_iter = TensorIteratorConfig()
                                       .add_output(weight)
                                       .add_input(weight)
                                       .add_input(grad)
                                       .check_all_same_dtype(true)
                                       .build();
  dpcpp_kernel_for_tensor_iter(
      update_iter, [=](float weight_elem, float adam_step_elem) -> float {
        weight_elem -= adam_step_elem * learning_rate * true_ratio;
        return weight_elem;
      });
}

template <typename scalar_t>
void launch_vec_kernel_Lamb_master_weight(
    Tensor& master_weight,
    Tensor& weight,
    Tensor& grad,
    Tensor& avg,
    Tensor& avg_sq,
    const int step,
    const double beta1,
    const double beta2,
    const double bias_correction1,
    const double bias_correction2,
    const double learning_rate,
    const double weight_decay,
    const double eps) {
  Tensor param_norm_tensor = at::zeros_like(master_weight);
  Tensor rtw_norm_tensor = at::zeros_like(master_weight);
  // for fp32, we reuse grad to store adam_step, but for half/bf16, precision
  // will loss, we need an fp32 tensor to restore this
  Tensor adam_step_tensor = at::zeros_like(master_weight);

  at::TensorIterator iter = TensorIteratorConfig()
                                .add_output(param_norm_tensor)
                                .add_output(rtw_norm_tensor)
                                .add_output(adam_step_tensor)
                                .add_output(avg)
                                .add_output(avg_sq)
                                .add_input(weight)
                                .add_input(grad)
                                .add_input(master_weight)
                                .add_input(avg)
                                .add_input(avg_sq)
                                .check_all_same_dtype(false)
                                .build();

  dpcpp_kernel_multiple_outputs_for_tensor_iter(
      iter,
      [=](scalar_t weight,
          scalar_t grad_elem,
          float master_weight_elem,
          float exp_avg_elem,
          float exp_avg_sq_elem)
          -> std::tuple<float, float, float, float, float> {
        grad_elem = static_cast<float>(grad_elem);
        exp_avg_elem = exp_avg_elem * beta1 + (1 - beta1) * grad_elem;
        exp_avg_sq_elem =
            exp_avg_sq_elem * beta2 + grad_elem * grad_elem * (1 - beta2);
        auto adam_step_elem = exp_avg_elem / bias_correction1 /
            (Numerics<float>::sqrt(exp_avg_sq_elem / bias_correction2) + eps);
        adam_step_elem = adam_step_elem + master_weight_elem * weight_decay;

        return std::tuple<float, float, float, float, float>{
            master_weight_elem * master_weight_elem,
            adam_step_elem * adam_step_elem,
            adam_step_elem,
            exp_avg_elem,
            exp_avg_sq_elem};
      });

  auto param_norm_sum = at::sum(param_norm_tensor).item().to<float>();
  auto rtw_norm_sum = at::sum(rtw_norm_tensor).item().to<float>();
  float true_ratio = std::sqrt(param_norm_sum) / std::sqrt(rtw_norm_sum);

  at::TensorIterator update_iter = TensorIteratorConfig()
                                       .add_output(weight)
                                       .add_output(master_weight)
                                       .add_input(weight)
                                       .add_input(master_weight)
                                       .add_input(adam_step_tensor)
                                       .check_all_same_dtype(false)
                                       .build();
  dpcpp_kernel_multiple_outputs_for_tensor_iter(
      update_iter,
      [=](scalar_t weight_elem,
          float master_weight_elem,
          float adam_step_elem) -> std::tuple<scalar_t, float> {
        weight_elem -= adam_step_elem * learning_rate * true_ratio;
        return std::tuple<scalar_t, float>{
            static_cast<scalar_t>(weight_elem), weight_elem};
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lamb_fused_step(
    const at::Tensor& param,
    const at::Tensor& exp_avg,
    const at::Tensor& exp_avg_sq,
    const at::Tensor& grad,
    const at::Tensor& param2,
    int64_t step,
    double beta1,
    double beta2,
    double learning_rate,
    double weight_decay,
    double eps) {
  RECORD_FUNCTION(
      "torch_ipex::lamb_fused_step", c10::ArrayRef<c10::IValue>({}));
  // need to keep align with cpu semantic, conver tensor to non-const here
  auto param_ = const_cast<at::Tensor&>(param);
  auto exp_avg_ = const_cast<at::Tensor&>(exp_avg);
  auto exp_avg_sq_ = const_cast<at::Tensor&>(exp_avg_sq);
  auto grad_ = const_cast<at::Tensor&>(grad);
  auto param2_ = const_cast<at::Tensor&>(param2);
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
  TORCH_CHECK(
      param2_.numel() == 0 || param_.sizes() == param2_.sizes(),
      "Expect param and param2_ have the same sizes, param sizes: ",
      param_.sizes(),
      "; param2_ sizes: ",
      param2_.sizes());

  // after inference, the model weight in the next training epoch maybe cached
  // in block layout, so to plain now if needed
  param_ = to_plain_if_needed_(param_);
  grad_ = to_plain_if_needed_(grad_);

  // support contiguous and channels_last contiguous
  auto memory_format = param_.suggest_memory_format();
  param_ = param_.contiguous(memory_format);
  exp_avg_ = exp_avg_.contiguous(memory_format);
  exp_avg_sq_ = exp_avg_sq_.contiguous(memory_format);
  grad_ = grad_.contiguous(memory_format);

  const auto beta1_value = static_cast<float>(beta1);
  const auto beta2_value = static_cast<float>(beta2);
  const auto learning_rate_value = static_cast<float>(learning_rate);
  const auto weight_decay_value = static_cast<float>(weight_decay);
  const auto eps_value = static_cast<float>(eps);
  const auto bias_correction1 = 1 - std::pow(beta1, step);
  const auto bias_correction2 = 1 - std::pow(beta2, step);

  // if param2_ has tensor we will go to master weight mode, where master_weight
  // is fp32 and param2_ is bf16 of fp16 (currently not support)
  if (param2_.numel()) {
    // master weight mode
    param2_ = to_plain_if_needed_(param2_);
    param2_ = param2_.contiguous(memory_format);
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        param2_.scalar_type(),
        "lamb_fused_step",
        [&]() {
          launch_vec_kernel_Lamb_master_weight<scalar_t>(
              param_,
              param2_,
              grad_,
              exp_avg_,
              exp_avg_sq_,
              step,
              beta1_value,
              beta2_value,
              bias_correction1,
              bias_correction2,
              learning_rate_value,
              weight_decay_value,
              eps_value);
        });
  } else {
    // normal mode, all float 32
    launch_vec_kernel_Lamb(
        param_,
        grad_,
        exp_avg_,
        exp_avg_sq_,
        step,
        beta1_value,
        beta2_value,
        bias_correction1,
        bias_correction2,
        learning_rate_value,
        weight_decay_value,
        eps_value);
  }
  return std::make_tuple(param_, exp_avg_, exp_avg_sq_);
}
} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "lamb_fused_step",
      at::AtenIpexTypeXPU::lamb_fused_step,
      c10::DispatchKey::XPU);
}
} // namespace
