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
#include "Loops.h"
#include "LoopsTemplates.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"

#include <aten/operators/MemoryAccess.h>
#include "utils/CustomOperatorRegistration.h"

using namespace xpu::dpcpp;
using namespace xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

// no master weight, all tensor are fp32
static void ComputeAdagradKernel(
    Tensor& weight,
    Tensor& grads,
    Tensor& state_sum,
    const bool use_weight_decay,
    const float learning_rate,
    const float weight_decay,
    const float lr_decay,
    const float eps_value,
    const int step) {
  // calculate current learing rate
  double clr = learning_rate / (1 + (step - 1) * lr_decay);

  at::TensorIterator iter = TensorIteratorConfig()
                                .add_output(weight)
                                .add_output(state_sum)
                                .add_input(weight)
                                .add_input(grads)
                                .add_input(state_sum)
                                .build();

  dpcpp_kernel_multiple_outputs_for_tensor_iter(
      iter,
      [=](float weight_elem,
          float grad_elem,
          float state_sum_elem) -> std::tuple<float, float> {
        if (use_weight_decay) {
          grad_elem += weight_elem * weight_decay;
        }
        state_sum_elem += grad_elem * grad_elem;

        float std_val = Numerics<float>::sqrt(state_sum_elem) + eps_value;

        weight_elem = weight_elem - grad_elem / std_val * clr;

        return std::tuple(weight_elem, state_sum_elem);
      });
}

template <typename scalar_t>
void ComputeAdagradKernelMasterWeight(
    Tensor& master_weight,
    Tensor& grad,
    Tensor& state_sum,
    Tensor& weight,
    const bool use_weight_decay,
    const double learning_rate,
    const double weight_decay,
    const double lr_decay,
    const double eps_value,
    const int step) {
  // calculate current learing rate
  double clr = learning_rate / (1 + (step - 1) * lr_decay);
  at::TensorIterator iter = TensorIteratorConfig()
                                .check_all_same_dtype(false)
                                .add_output(weight)
                                .add_output(master_weight)
                                .add_output(state_sum)
                                .add_input(weight)
                                .add_input(master_weight)
                                .add_input(grad)
                                .add_input(state_sum)
                                .build();

  dpcpp_kernel_multiple_outputs_for_tensor_iter(
      iter,
      [=](scalar_t weight_elem,
          float master_weight_elem,
          scalar_t grad_elem,
          float state_sum_elem) -> std::tuple<scalar_t, float, float> {
        float fp32_grad_elem = static_cast<float>(grad_elem);

        if (use_weight_decay) {
          fp32_grad_elem += master_weight_elem * weight_decay;
        }
        state_sum_elem += fp32_grad_elem * fp32_grad_elem;

        float std_val = Numerics<float>::sqrt(state_sum_elem) + eps_value;

        master_weight_elem =
            master_weight_elem - fp32_grad_elem / std_val * clr;

        return std::tuple(
            static_cast<scalar_t>(master_weight_elem),
            master_weight_elem,
            state_sum_elem);
      });
}

} // namespace impl

std::tuple<at::Tensor, at::Tensor> adagrad_fused_step(
    const at::Tensor& param_,
    const at::Tensor& grad_,
    const at::Tensor& state_sum_,
    const at::Tensor& param2_,
    double step,
    double learning_rate,
    double weight_decay,
    double lr_decay,
    double eps) {
  RECORD_FUNCTION(
      "torch_ipex::adagrad_fused_step", c10::ArrayRef<c10::IValue>({}));
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

  const auto step_value = static_cast<float>(step);
  const auto learning_rate_value = static_cast<float>(learning_rate);
  const auto lr_decay_value = static_cast<float>(lr_decay);
  const auto weight_decay_value = static_cast<float>(weight_decay);
  const auto eps_value = static_cast<float>(eps);

  const OptionalDeviceGuard device_guard(device_of(param_));
  // after inference, the model weight in the next training epoch maybe cached
  // in block layout, so to plain now if needed

  auto param = to_plain_if_needed(param_);
  auto grad = to_plain_if_needed(grad_);

  bool use_weight_decay = false;
  if (weight_decay_value != 0) {
    use_weight_decay = true;
  }

  // convert memory format
  auto memory_format = param_.suggest_memory_format();
  param = param.contiguous(memory_format);
  grad = grad.contiguous(memory_format);
  auto state_sum = state_sum_.contiguous(memory_format);

  if (param2_.numel()) {
    // will use master weight
    auto param2 = to_plain_if_needed(param2_);
    param2 = param2.contiguous(memory_format);
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        param2_.scalar_type(),
        "adagrad_fused_step",
        [&] {
          impl::ComputeAdagradKernelMasterWeight<scalar_t>(
              param,
              grad,
              state_sum,
              param2,
              use_weight_decay,
              learning_rate_value,
              weight_decay_value,
              lr_decay_value,
              eps_value,
              step_value);
        });
  } else {
    // normal mode, all use fp32
    impl::ComputeAdagradKernel(
        param,
        grad,
        state_sum,
        use_weight_decay,
        learning_rate_value,
        weight_decay_value,
        lr_decay,
        eps_value,
        step_value);
  }
  return std::make_tuple(param, state_sum);
}

} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "adagrad_fused_step",
      at::AtenIpexTypeXPU::adagrad_fused_step,
      c10::DispatchKey::XPU);
}
} // namespace