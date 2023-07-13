#include <ATen/ATen.h>
#include <core/detail/IndexUtils.h>
#include <oneDNN/oneDNN.h>
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
    double momentum_,
    int64_t N) {
  auto iter = TensorIteratorConfig()
                  .add_output(running_mean)
                  .add_output(running_var)
                  .add_input(save_mean)
                  .add_input(save_var)
                  .add_input(running_mean)
                  .add_input(running_var)
                  .check_all_same_dtype(false)
                  .promote_inputs_to_common_dtype(false)
                  .build();

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      running_mean.scalar_type(),
      "batch_norm_update_stats",
      [&] {
        using acc_t = acc_type<scalar_t>;
        const auto bessel_correction_factor = static_cast<acc_t>(
            static_cast<double>(N) / static_cast<double>(N - 1));
        const auto momentum = static_cast<acc_t>(momentum_);

        dpcpp_kernel_multiple_outputs_for_tensor_iter(
            iter,
            [=](acc_t mean,
                acc_t var,
                scalar_t running_mean,
                scalar_t running_var) -> std::tuple<scalar_t, scalar_t> {
              const auto unbiased_var = var * bessel_correction_factor;
              return std::tuple<scalar_t, scalar_t>{
                  mean * momentum + (1 - momentum) * running_mean,
                  unbiased_var * momentum + (1 - momentum) * running_var,
              };
            });
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
    ss << "batch_norm backend got unsupported type=" << input.scalar_type();
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
    ss << "batch_norm backend got unsupported type=" << input.scalar_type();
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
    ss << "batch_norm backend got unsupported type=" << input.scalar_type();
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

template <typename scalar_t, typename accscalar_t, typename index_t>
std::tuple<Tensor, Tensor> batch_norm_gather_stats_xpu_template(
    const Tensor& mean_,
    const Tensor& invstd_,
    const Tensor& running_mean_,
    const Tensor& running_var_,
    double momentum,
    double epsilon,
    const Tensor& counts_) {
  Tensor save_mean_;
  Tensor save_invstd_;

  auto features = mean_.size(1);
  auto input_options = mean_.options();
  if (mean_.scalar_type() == at::ScalarType::Half ||
      mean_.scalar_type() == at::ScalarType::BFloat16) {
    input_options = input_options.dtype(ScalarType::Float);
  }
  save_mean_ = at::empty({features}, input_options);
  save_invstd_ = at::empty({features}, input_options);

  auto mean = mean_.accessor<accscalar_t, 2>();
  auto invstd = invstd_.accessor<accscalar_t, 2>();
  auto running_mean =
      running_mean_.defined() ? running_mean_.data_ptr<scalar_t>() : nullptr;
  auto running_var =
      running_var_.defined() ? running_var_.data_ptr<scalar_t>() : nullptr;
  auto counts = counts_.data_ptr<scalar_t>();
  auto save_mean = save_mean_.data_ptr<accscalar_t>();
  auto save_invstd = save_invstd_.data_ptr<accscalar_t>();

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const auto wgroup_size = dpcppMaxWorkGroupSize(dev_id);
  const auto ngroups = (features + wgroup_size - 1) / wgroup_size;

  int world_size = mean_.size(0);
  // Avoid double issues in ATSM
  float momentum_ = momentum;
  float epsilon_ = epsilon;

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(ngroups * wgroup_size, wgroup_size),
        [=](sycl::nd_item<1> itemId) {
          auto tid = itemId.get_global_linear_id();

          // first the reductions each thread does separately
          if (tid < features) {
            accscalar_t avg = 0;
            accscalar_t var_n = 0;
            index_t n = 0;
            for (int j = 0; j < world_size; j++) {
              scalar_t count = counts[j];
              accscalar_t m = mean[j][tid];
              accscalar_t v = accscalar_t(1.0f) / (invstd[j][tid]);
              v = (v * v - epsilon_) * count;
              accscalar_t factor = 1.0f / (n + count);
              var_n += v + (avg - m) * (avg - m) * n * count * factor;
              avg = n * factor * avg + count * factor * m;
              n += count;
            }
            save_mean[tid] = avg;
            save_invstd[tid] = static_cast<accscalar_t>(1) /
                Numerics<accscalar_t>::sqrt(var_n / n + epsilon_);
            if (running_mean != nullptr) {
              running_mean[tid] = static_cast<scalar_t>(
                  (1 - momentum_) * running_mean[tid] + momentum_ * avg);
            }
            accscalar_t unbiasedVar = var_n / (n - 1);
            if (running_var != nullptr) {
              running_var[tid] = static_cast<scalar_t>(
                  (1 - momentum_) * running_var[tid] + momentum_ * unbiasedVar);
            }
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
  return std::make_tuple(save_mean_, save_invstd_);
}

std::tuple<Tensor, Tensor> batch_norm_gather_stats_with_counts_xpu(
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& running_mean_opt /* optional */,
    const c10::optional<Tensor>& running_var_opt /* optional */,
    double momentum,
    double epsilon,
    const Tensor& counts) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> running_mean_maybe_owned =
      at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  const Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return Tensor(); });

  auto scalar_type =
      running_mean.defined() ? running_mean.scalar_type() : self.scalar_type();
  return IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scalar_type,
      "batch_norm_update_stats_xpu",
      [&] {
        using accscalar_t = acc_type<scalar_t>;
        if (xpu::dpcpp::detail::canUse32BitIndexMath(self)) {
          return batch_norm_gather_stats_xpu_template<
              scalar_t,
              accscalar_t,
              int32_t>(
              mean,
              invstd,
              running_mean,
              running_var,
              momentum,
              epsilon,
              counts);
        } else {
          return batch_norm_gather_stats_xpu_template<
              scalar_t,
              accscalar_t,
              int64_t>(
              mean,
              invstd,
              running_mean,
              running_var,
              momentum,
              epsilon,
              counts);
        }
      });
}

// accepting input(self) here to determine template data types, since
// running_mean/running_var are optional
std::tuple<Tensor, Tensor> batch_norm_gather_stats(
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    double momentum,
    double epsilon,
    int64_t count) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> running_mean_maybe_owned =
      at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  const Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return Tensor(); });

  Tensor counts_ = at::empty(
      mean.size(0),
      self.options().dtype(
          running_mean.defined() ? running_mean.dtype() : self.dtype()));
  counts_.fill_(count);
  return batch_norm_gather_stats_with_counts_xpu(
      self,
      mean,
      invstd,
      running_mean,
      running_var,
      momentum,
      epsilon,
      counts_);
}

std::tuple<Tensor, Tensor> batch_norm_gather_stats_with_counts(
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& running_mean_opt /* optional */,
    const c10::optional<Tensor>& running_var_opt /* optional */,
    double momentum,
    double epsilon,
    const Tensor& counts) {
  // See [Note: hacky wrapper removal for optional tensor]
  return batch_norm_gather_stats_with_counts_xpu(
      self,
      mean,
      invstd,
      running_mean_opt,
      running_var_opt,
      momentum,
      epsilon,
      counts);
}

} // namespace AtenIpexTypeXPU
} // namespace at
