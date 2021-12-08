#pragma once

#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace torch_ipex {
namespace cpu {
// at::TensorAccessor when it is defined to work around undefined...
template <typename scalar_t>
static at::TensorAccessor<scalar_t, 1> conditional_accessor_1d(
    const at::Tensor& t) {
  if (!t.defined()) {
    return at::TensorAccessor<scalar_t, 1>(nullptr, nullptr, nullptr);
  }
  return t.accessor<scalar_t, 1>();
}

template <typename scalar_t>
static scalar_t* conditional_data_ptr(const at::Tensor& t) {
  return t.defined() ? t.contiguous().data_ptr<scalar_t>() : nullptr;
}

inline at::ScalarType first_type() {
  return at::ScalarType::Undefined;
}

template <typename... Args>
inline at::ScalarType first_type(
    const at::Tensor& arg,
    const Args&... parameters) {
  return arg.defined() ? arg.scalar_type() : first_type(parameters...);
}

template <typename... Args>
inline bool is_mixed_type(const at::Tensor& input, const Args&... parameters) {
  const auto parameter_type = first_type(parameters...);
  return (
      (parameter_type != at::ScalarType::Undefined) &&
      (parameter_type != input.scalar_type()));
}

inline void checkMixedDataTypes(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd) {
  // At the moment, the only allowed mixed dtype pattern: input(bfloat16) +
  // weight/bias(float)
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::BFloat16,
      "BatchNorm (CPU) with mixed dtype: expect input to have scalar "
      "type of at::BFloat16");
  TORCH_CHECK(
      !weight.defined() || weight.scalar_type() == at::ScalarType::Float,
      "BatchNorm (CPU) with mixed dtype: expect weight either "
      "undefined or have scalar type of Float");
  TORCH_CHECK(
      !bias.defined() || bias.scalar_type() == at::ScalarType::Float,
      "BatchNorm (CPU) with mixed dtype: expect bias either undefined "
      "or have scalar type of Float");
  TORCH_CHECK(
      !running_mean.defined() ||
          running_mean.scalar_type() == at::ScalarType::Float,
      "BatchNorm (CPU) with mixed dtype: expect running_mean either "
      "undefined or have scalar type of Float");
  TORCH_CHECK(
      !running_var.defined() ||
          running_var.scalar_type() == at::ScalarType::Float,
      "BatchNorm (CPU) with mixed dtype: expect running_var either "
      "undefined or have scalar type of Float");
  TORCH_CHECK(
      !save_mean.defined() || save_mean.scalar_type() == at::ScalarType::Float,
      "BatchNorm (CPU) with mixed dtype: expect save_mean either "
      "undefined or have scalar type of Float");
  TORCH_CHECK(
      !save_invstd.defined() ||
          save_invstd.scalar_type() == at::ScalarType::Float,
      "BatchNorm (CPU) with mixed dtype: expect save_invstd either "
      "undefined or have scalar type of Float");
}

// use float for bfloat16 accumulation
template <typename scalar_t>
struct ParamAccType {
  using type = scalar_t;
};
template <>
struct ParamAccType<at::BFloat16> {
  using type = float;
};

template <typename scalar_t>
using param_acc_t = typename ParamAccType<scalar_t>::type;

inline at::TensorOptions param_options(const at::Tensor& input) {
  if (input.scalar_type() == at::ScalarType::BFloat16) {
    return input.options().dtype(at::kFloat);
  } else {
    return input.options();
  }
}

template <typename param_t, typename param2_t>
void batch_norm_cpu_collect_linear_and_constant_terms(
    param2_t* alpha,
    param2_t* beta,
    int64_t n_channel,
    const at::Tensor& weight /* optional */,
    const at::Tensor& bias /* optional */,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double eps);

template <typename scalar_t, typename param_t>
void batch_norm_cpu_contiguous_impl(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double eps);

template <typename scalar_t, typename param_t>
void batch_norm_cpu_channels_last_impl(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    const at::Tensor& running_mean,
    const at::Tensor& runnning_var,
    bool train,
    double eps);

template <typename scalar_t, typename param_t>
void batch_norm_cpu_collect_stats_contiguous_impl(
    at::Tensor& mean,
    at::Tensor& var_sum,
    const at::Tensor& input);

template <typename scalar_t, typename param_t>
void batch_norm_cpu_collect_stats_channels_last_impl(
    at::Tensor& mean,
    at::Tensor& var_sum,
    const at::Tensor& input);

template <typename scalar_t, typename param_t>
void batch_norm_cpu_backward_contiguous_impl(
    at::Tensor& grad_input,
    at::Tensor& grad_weight,
    at::Tensor& grad_bias,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    bool train,
    double eps);

template <typename scalar_t, typename param_t>
void batch_norm_cpu_backward_channels_last_impl(
    at::Tensor& grad_input,
    at::Tensor& grad_weight,
    at::Tensor& grad_bias,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    bool train,
    double eps);

void batch_norm_cpu_kernel(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double eps);

void batch_norm_cpu_collect_stats_kernel(
    at::Tensor& mean,
    at::Tensor& var_sum,
    const at::Tensor& input);

void batch_norm_cpu_backward_kernel(
    at::Tensor& grad_input,
    at::Tensor& grad_weight,
    at::Tensor& grad_bias,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    bool train,
    double eps);

template <typename scalar_t, typename param_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor>
batch_norm_cpu_transform_input_template(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& save_mean /* optional */,
    const at::Tensor& save_invstd /* optional */,
    const at::Tensor& running_mean /* optional */,
    const at::Tensor& running_var /* optional */,
    bool train,
    double eps);

template <
    typename scalar_t,
    typename param_t,
    template <typename T>
    class VarTransform>
std::tuple<at::Tensor, at::Tensor> batch_norm_cpu_update_stats_template(
    const at::Tensor& input,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    double momentum,
    double eps);

template <typename scalar_t, typename param_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_backward_cpu_template(
    const at::Tensor& grad_out_,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    bool train,
    double eps,
    std::array<bool, 3> grad_input_mask);

std::tuple<at::Tensor, at::Tensor> batch_norm_update_stats_cpu(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    double momentum);

std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_cpu(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_backward_cpu(
    const at::Tensor& grad_out,
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    const c10::optional<at::Tensor>& save_mean_opt,
    const c10::optional<at::Tensor>& save_invstd_opt,
    bool train,
    double eps,
    std::array<bool, 3> grad_input_mask);

} // namespace cpu
} // namespace torch_ipex
