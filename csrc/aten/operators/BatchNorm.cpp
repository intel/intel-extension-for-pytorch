#include <ATen/ATen.h>

#include <oneDNN/oneDNN.h>

using namespace dnnl;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

static inline at::Tensor condition_contiguous(const at::Tensor& t) {
  if (!t.defined()) {
    return t;
  }

  if (t.defined() && !is_smf_channels_last(t)) {
    return t.contiguous();
  }

  // if (t.defined() && is_smf_channels_last(t))
  return t.contiguous(get_cl_tag_by_ndim(t.ndimension()));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool training,
    double momentum,
    double epsilon) {
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
  } else {
    return xpu::oneDNN::batch_normalization(
        condition_contiguous(input),
        condition_contiguous(weight),
        condition_contiguous(bias),
        condition_contiguous(running_mean),
        condition_contiguous(running_var),
        training,
        momentum,
        epsilon);
  }
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
  TORCH_WARN_ONCE(
      "Warning: the function declaration is a little different, see in RegisterXPU.cpp. \n(const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask) \n");

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

} // namespace AtenIpexTypeXPU
} // namespace at
