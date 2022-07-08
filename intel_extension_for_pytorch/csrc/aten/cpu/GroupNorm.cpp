#include "GroupNorm.h"
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <c10/util/accumulate.h>
#include "csrc/utils/ipex_op_profile.h"
#include "csrc/utils/library.h"

#include <array>
#include <functional>
#include <numeric>
#include <tuple>
#include <vector>
#include "utils/utils.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(GroupNormKernel);
DEFINE_DISPATCH(GroupNormBackwardKernel);

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm(
    const at::Tensor& X,
    const c10::optional<at::Tensor>& gamma_opt /* optional */,
    const c10::optional<at::Tensor>& beta_opt /* optional */,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::native_group_norm\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::native_group_norm", c10::ArrayRef<c10::IValue>({}));

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const at::Tensor& gamma = *gamma_maybe_owned;
  const at::Tensor& beta =
      c10::value_or_else(beta_opt, [] { return at::Tensor(); });

  auto memory_format = X.device().is_cpu() ? X.suggest_memory_format()
                                           : at::MemoryFormat::Contiguous;

  at::Tensor Y;
  // Add channels last 1d input support
  if (is_channels_last_1d(X)) {
    Y = at::native::empty_like(X);
  } else {
    Y = at::native::empty_like(
        X,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        memory_format);
  }
  at::Tensor mean = at::empty({N, group}, X.options());
  at::Tensor rstd = at::empty({N, group}, X.options());
  GroupNormKernel(
      X.device().type(), X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
  return std::make_tuple(Y, mean, rstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_backward(
    const at::Tensor& dY,
    const at::Tensor& X,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& gamma_opt,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    std::array<bool, 3> grad_input_mask) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::native_group_norm_backward\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::native_group_norm_backward", c10::ArrayRef<c10::IValue>({}));

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const at::Tensor& gamma = *gamma_maybe_owned;

  at::Tensor dX;
  at::Tensor dgamma;
  at::Tensor dbeta;
  if (grad_input_mask[0]) {
    dX = at::native::empty_like(
        X,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[1]) {
    dgamma = at::native::empty_like(
        gamma,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[2]) {
    dbeta = at::native::empty_like(
        gamma,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  }
  GroupNormBackwardKernel(
      X.device().type(),
      dY,
      X,
      mean,
      rstd,
      gamma,
      N,
      C,
      HxW,
      group,
      dX,
      dgamma,
      dbeta);
  return std::make_tuple(dX, dgamma, dbeta);
}

at::Tensor group_norm(
    const at::Tensor& input,
    int64_t num_groups,
    const c10::optional<at::Tensor>& weight_opt /* optional */,
    const c10::optional<at::Tensor>& bias_opt /* optional */,
    double eps,
    bool /* cudnn_enabled, deprecated */) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::group_norm\n");
#endif
  IPEX_RECORD_FUNCTION(
      "torch_ipex::group_norm", c10::ArrayRef<c10::IValue>({}));

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& weight = *weight_maybe_owned;
  const at::Tensor& bias =
      c10::value_or_else(bias_opt, [] { return at::Tensor(); });

  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  TORCH_CHECK(
      C % num_groups == 0,
      "Expected number of channels in input to be divisible by ",
      "num_groups, but got input of shape ",
      input.sizes(),
      " and "
      "num_groups=",
      num_groups);
  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && weight.numel() == C),
      "Expected weight to be a vector of size equal to the number of ",
      "channels in input, but got weight of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());
  TORCH_CHECK(
      !bias.defined() || (bias.dim() == 1 && bias.numel() == C),
      "Expected bias to be a vector of size equal to the number of ",
      "channels in input, but got bias of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());

  const auto input_shape = input.sizes();
  const int64_t HxW =
      c10::multiply_integers(input_shape.cbegin() + 2, input_shape.cend());

  const at::Tensor kEmpty;
  // Add channels last 1d input support
  auto memory_format = input.suggest_memory_format();
  const auto& X = input.device().is_cpu()
      ? (is_channels_last_1d(input) ? input : input.contiguous(memory_format))
      : input.contiguous();
  const auto& gamma = weight.defined()
      ? (is_channels_last_1d(weight) ? weight : weight.contiguous())
      : kEmpty;
  const auto& beta = bias.defined()
      ? (is_channels_last_1d(bias) ? bias : bias.contiguous())
      : kEmpty;
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  return std::get<0>(
      at::native_group_norm(X, gamma, beta, N, C, HxW, num_groups, eps));
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::group_norm"),
      TORCH_FN((&torch_ipex::cpu::group_norm)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::native_group_norm"),
      TORCH_FN((&torch_ipex::cpu::native_group_norm)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::native_group_norm_backward"),
      TORCH_FN((&torch_ipex::cpu::native_group_norm_backward)));
}

} // namespace cpu
} // namespace torch_ipex