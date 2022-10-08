#include <ATen/ATen.h>
#include <oneDNN/oneDNN.h>
#include <oneapi/dpl/tuple>
#include "Loops.h"
#include "Resize.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"
using namespace dnnl;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

static inline at::Tensor condition_contiguous(const at::Tensor& t) {
  auto ndim = t.ndimension();
  if (!t.defined()) {
    return t;
  }

  if (t.defined() && !is_smf_channels_last(t)) {
    return t.contiguous();
  }

  // if (t.defined() && is_smf_channels_last(t))
  auto cl_tag = get_cl_tag_by_ndim(t.ndimension());
  if (CHANNELSLAST1D_DPCPP == cl_tag) {
    auto tmp = t.contiguous();
    return convert_tensor_to_channels_last_1d(tmp);
  }

  return t.contiguous(cl_tag);
}

void batch_norm_update_stats(
    const Tensor& save_mean,
    const Tensor& save_var,
    const Tensor& running_mean,
    const Tensor& running_var,
    double momentum,
    int64_t N) {
  auto feature_num = N;
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      running_mean.scalar_type(),
      "mScale1",
      [&]() {
        dpcppMemoryScale1(
            running_mean.data_ptr<scalar_t>(),
            save_mean.data_ptr<float>(),
            feature_num,
            momentum);
      });
  size_t orig_size = feature_num;
  size_t adjust_size = orig_size - 1;
  float adjust_factor = (static_cast<float>(orig_size)) / adjust_size;
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      running_var.scalar_type(),
      "mScale2",
      [&]() {
        dpcppMemoryScale2(
            running_var.data_ptr<scalar_t>(),
            save_var.data_ptr<float>(),
            feature_num,
            adjust_factor,
            momentum);
      });
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> native_batch_norm_out(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool training,
    double momentum,
    double epsilon,
    at::Tensor& out,
    at::Tensor& save_mean,
    at::Tensor& save_invstd) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  c10::MaybeOwned<Tensor> running_mean_maybe_owned =
      at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;

  c10::MaybeOwned<Tensor> running_var_maybe_owned =
      at::borrow_from_optional_tensor(running_var_opt);
  const Tensor& running_var = *running_var_maybe_owned;

  if (running_mean.defined() && running_var.defined()) {
    checkBackend(
        "batch_norm",
        {input, weight, bias, running_mean, running_var},
        Backend::XPU);
  } else {
    checkBackend("batch_norm", {input, weight, bias}, Backend::XPU);
  }

  if (input.scalar_type() != at::ScalarType::Float &&
      input.scalar_type() != at::ScalarType::Half &&
      input.scalar_type() != at::ScalarType::BFloat16) {
    std::stringstream ss;
    ss << "DPCPP batch_norm backend got unsupported type="
       << input.scalar_type();
    TORCH_CHECK(0, ss.str());
  }

  xpu::oneDNN::batch_normalization(
      condition_contiguous(input),
      condition_contiguous(weight),
      condition_contiguous(bias),
      condition_contiguous(running_mean),
      condition_contiguous(running_var),
      training,
      momentum,
      epsilon,
      out,
      save_mean,
      save_invstd);

  // Update running_mean and running_var
  if (training && running_mean.defined() && running_var.defined()) {
    const int64_t N = input.size(1);
    batch_norm_update_stats(
        save_mean, save_invstd, running_mean, running_var, momentum, N);
  }

  return std::tuple<Tensor&, Tensor&, Tensor&>(out, save_mean, save_invstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm(
    const at::Tensor& src,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool training,
    double momentum,
    double epsilon) {
  // Make empty tensors. Those empty tensor will be initialized in
  // oneDNN::batch_norm

  Tensor dst;
  Tensor save_mean;
  Tensor save_var;

  native_batch_norm_out(
      src,
      weight_opt,
      bias_opt,
      running_mean_opt,
      running_var_opt,
      training,
      momentum,
      epsilon,
      dst,
      save_mean,
      save_var);
  return std::tuple<Tensor&, Tensor&, Tensor&>(dst, save_mean, save_var);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    const c10::optional<at::Tensor>& save_mean_opt,
    const c10::optional<at::Tensor>& save_var_opt,
    bool training,
    double epsilon,
    std::array<bool, 3> grad_input_mask) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  c10::MaybeOwned<Tensor> running_mean_maybe_owned =
      at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;

  c10::MaybeOwned<Tensor> running_var_maybe_owned =
      at::borrow_from_optional_tensor(running_var_opt);
  const Tensor& running_var = *running_var_maybe_owned;

  c10::MaybeOwned<Tensor> save_mean_maybe_owned =
      at::borrow_from_optional_tensor(save_mean_opt);
  const Tensor& save_mean = *save_mean_maybe_owned;

  c10::MaybeOwned<Tensor> save_var_maybe_owned =
      at::borrow_from_optional_tensor(save_var_opt);
  const Tensor& save_var = *save_var_maybe_owned;

  if (save_mean.defined() && save_var.defined()) {
    checkBackend(
        "batch_norm",
        {input, weight, grad_output, save_mean, save_var},
        Backend::XPU);
  } else {
    checkBackend("batch_norm", {input, weight, grad_output}, Backend::XPU);
  }

  if (input.scalar_type() != at::ScalarType::Float &&
      input.scalar_type() != at::ScalarType::Half &&
      input.scalar_type() != at::ScalarType::BFloat16) {
    std::stringstream ss;
    ss << "DPCPP batch_norm backend got unsupported type="
       << input.scalar_type();
    TORCH_CHECK(0, ss.str());
  } else {
    return xpu::oneDNN::batch_normalization_backward(
        condition_contiguous(grad_output),
        condition_contiguous(input),
        condition_contiguous(weight),
        condition_contiguous(running_mean),
        condition_contiguous(running_var),
        condition_contiguous(save_mean),
        condition_contiguous(save_var),
        training,
        epsilon,
        grad_input_mask);
  }
}

// For native_batch_norm, we don't use this batch_norm_elemt_out
// Because oneDNN could handle it automatically.
at::Tensor& batch_norm_elemt_out(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double eps,
    at::Tensor& out) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  checkBackend("batch_norm", {input, weight, bias, mean, invstd}, Backend::XPU);

  // The check follows native batch norm
  if (input.scalar_type() != at::ScalarType::Float &&
      input.scalar_type() != at::ScalarType::Half &&
      input.scalar_type() != at::ScalarType::BFloat16) {
    std::stringstream ss;
    ss << "DPCPP batch_norm backend got unsupported type="
       << input.scalar_type();
    TORCH_CHECK(0, ss.str());
  }

  // Don't need these two, thus use dummy tensor.
  // In current stat, the oneDNN batch norm flag should be
  // inference + use_global_stats.
  Tensor dummy_mean;
  Tensor dummy_var;

  // don't need momentum, epsilon, thus use dummy data
  xpu::oneDNN::batch_normalization(
      condition_contiguous(input),
      condition_contiguous(weight),
      condition_contiguous(bias),
      condition_contiguous(mean),
      condition_contiguous(invstd),
      /* training*/ false,
      /* momentum */ 1.0,
      /*epsilon , dummy epsilon*/ 1e-5,
      out,
      dummy_mean,
      dummy_var);
  return out;
}

at::Tensor batch_norm_elemt(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double eps) {
  // Empty tensor, it will be initialized in batch_norm_elemt_out
  Tensor out;
  batch_norm_elemt_out(input, weight, bias, mean, invstd, eps, out);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
