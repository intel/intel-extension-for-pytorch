#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/record_function.h>

#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include "Normalization.h"

#include <vector>

#include "csrc/utils/library.h"

static const int MIOPEN_DIM_MAX = 5;

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(batch_norm_cpu_kernel_stub);
DEFINE_DISPATCH(batch_norm_cpu_collect_stats_kernel_stub);
DEFINE_DISPATCH(batch_norm_cpu_backward_kernel_stub);

void check_dims_match_num_input_features(
    const char* arg_name,
    int64_t expected,
    int64_t actual) {
  TORCH_CHECK(
      actual == expected,
      arg_name,
      " should contain ",
      expected,
      " elements not ",
      actual);
}

static inline at::Tensor repeat_if_defined(
    const at::Tensor& t,
    int64_t repeat) {
  if (t.defined()) {
    return t.repeat(repeat);
  }
  return t;
}

template <typename T>
struct InvStd {
  T operator()(T var, double epsilon) const {
    T invstd = 0;
    if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
      invstd = static_cast<T>(1) / std::sqrt(var + epsilon);
    }
    return invstd;
  }
};

template <typename T>
struct Var {
  T operator()(T var, double epsilon) const {
    return var;
  }
};

static inline bool is_contiguous(const at::Tensor& t) {
  return t.is_contiguous() || t.is_contiguous(at::MemoryFormat::ChannelsLast);
}

// For some ambiguous cases, it is possible a channels last contiguous
// at::Tensor has
//   `suggest_memory_format` of Contiguous.
// See https://github.com/pytorch/pytorch/issues/63224 for details.
static inline at::MemoryFormat suggest_memory_format_contig(
    const at::Tensor& t) {
  return t.is_contiguous() ? at::MemoryFormat::Contiguous
                           : at::MemoryFormat::ChannelsLast;
}

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
    double eps) {
  bool all_contiguous = is_contiguous(input) &&
      (!weight.defined() || weight.is_contiguous()) &&
      (!bias.defined() || bias.is_contiguous()) &&
      running_mean.is_contiguous() && running_var.is_contiguous();

  // inference contiguous path
  if (all_contiguous) {
    at::Tensor output =
        at::empty_like(input, suggest_memory_format_contig(input));

#if defined(DYN_DISP_BUILD)
    batch_norm_cpu_kernel_stub(
        kCPU,
        output,
        input,
        weight,
        bias,
        save_mean,
        save_invstd,
        running_mean,
        running_var,
        train,
        eps);
#else
    batch_norm_cpu_kernel_impl(
        output,
        input,
        weight,
        bias,
        save_mean,
        save_invstd,
        running_mean,
        running_var,
        train,
        eps);
#endif

    return std::make_tuple(output, save_mean, save_invstd);
  }

  const int64_t ndim = input.dim();
  // Helper to convert 1d tensors to an nd tensor that broadcasts with input
  // All elements go into the channel dimension
  at::DimVector sizes(ndim, 1), strides(ndim, 0);
  auto as_nd = [&](const at::Tensor& t) {
    TORCH_INTERNAL_ASSERT(t.defined() && t.dim() == 1);
    sizes[1] = t.sizes()[0];
    strides[1] = t.strides()[0];
    return t.as_strided(sizes, strides);
  };

  auto mean = as_nd(train ? save_mean : running_mean);
  auto invstd = as_nd([&] {
    if (train) {
      return save_invstd;
    } else {
      return 1 / at::sqrt(running_var + eps);
    }
  }());
  const bool mixed_type = !std::is_same<scalar_t, param_t>::value;
  auto w = weight.defined()
      ? as_nd(weight)
      : at::detail::scalar_tensor_static(
            1, mixed_type ? at::kFloat : input.scalar_type(), at::kCPU);
  auto b = bias.defined()
      ? as_nd(bias)
      : at::detail::scalar_tensor_static(
            0, mixed_type ? at::kFloat : input.scalar_type(), at::kCPU);

  at::Tensor output = at::empty_like(input, input.suggest_memory_format());
  auto iter = at::TensorIteratorConfig()
                  .add_output(output)
                  .add_input(input)
                  .add_input(mean)
                  .add_input(invstd)
                  .add_input(w)
                  .add_input(b)
                  .check_all_same_dtype(false)
                  .promote_inputs_to_common_dtype(false)
                  .build();

  at::native::cpu_kernel(
      iter,
      [=](scalar_t input,
          param_t mean,
          param_t invstd,
          param_t weight,
          param_t bias) { return ((input - mean) * invstd) * weight + bias; });
  return std::make_tuple(output, save_mean, save_invstd);
}

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
    double eps) {
  using accscalar_t = at::acc_type<scalar_t, false>;

  int64_t n_input = input.size(1);
  int64_t n = input.numel() / n_input;
  const int64_t ndim = input.dim();

  auto running_mean_a = conditional_accessor_1d<param_t>(running_mean);
  auto running_var_a = conditional_accessor_1d<param_t>(running_var);

  const bool mixed_type = !std::is_same<scalar_t, param_t>::value;
  const auto dtype = mixed_type ? at::kFloat : input.scalar_type();

  bool all_contiguous = is_contiguous(input);

  // Reduce all dimensions except dim=1
  at::DimVector reduce_dims(ndim - 1);
  reduce_dims[0] = 0;
  for (int64_t i = 2; i < ndim; ++i) {
    reduce_dims[i - 1] = i;
  }

  // For contiguous case, leave 'mean' computation to kernel
  at::Tensor save_mean = all_contiguous
      ? at::empty({n_input}, input.options().dtype(dtype))
      : at::mean(input, /*dims=*/reduce_dims, /*keepdim=*/false, dtype);
  at::Tensor save_var_transform =
      at::empty({n_input}, input.options().dtype(dtype));
  auto save_mean_a = save_mean.accessor<param_t, 1>();
  auto save_var_transform_a = save_var_transform.accessor<param_t, 1>();

  if (all_contiguous) {
    auto _mean = at::empty({n_input}, input.options().dtype(dtype));
    auto _var_sum = at::empty({n_input}, input.options().dtype(dtype));
    auto _mean_a = _mean.accessor<param_t, 1>();
    auto _var_sum_a = _var_sum.accessor<param_t, 1>();

#if defined(DYN_DISP_BUILD)
    batch_norm_cpu_collect_stats_kernel_stub(kCPU, _mean, _var_sum, input);
#else
    batch_norm_cpu_collect_stats_kernel_impl(_mean, _var_sum, input);
#endif

    at::parallel_for(0, n_input, 1, [&](int64_t b_begin, int64_t b_end) {
      for (int64_t f = b_begin; f < b_end; ++f) {
        save_mean_a[f] = _mean_a[f];
        save_var_transform_a[f] =
            VarTransform<accscalar_t>{}(_var_sum_a[f] / n, eps);

        if (running_mean.defined()) {
          running_mean_a[f] =
              momentum * _mean_a[f] + (1 - momentum) * running_mean_a[f];
        }
        if (running_var.defined()) {
          accscalar_t unbiased_var = _var_sum_a[f] / (n - 1);
          running_var_a[f] =
              momentum * unbiased_var + (1 - momentum) * running_var_a[f];
        }
      }
    });

    return std::make_tuple(save_mean, save_var_transform);
  }

  // non-contiguous path
  auto channel_stride = input.strides()[1];
  auto in_data = input.data_ptr<scalar_t>();
  auto reduce_iter = at::TensorIteratorConfig()
                         .add_input(input)
                         .resize_outputs(false)
                         .declare_static_shape(input.sizes(), /*squash_dims=*/1)
                         .check_all_same_dtype(false)
                         .promote_inputs_to_common_dtype(false)
                         .build();

  at::parallel_for(0, n_input, 1, [&](int64_t b_begin, int64_t b_end) {
    at::TensorIterator iter(reduce_iter);
    for (int64_t f = b_begin; f < b_end; ++f) {
      // compute variance per input
      iter.unsafe_replace_operand(0, in_data + channel_stride * f);
      accscalar_t var_sum = 0;
      auto mean = static_cast<accscalar_t>(save_mean_a[f]);
      at::native::cpu_serial_kernel(iter, [&](const scalar_t i) -> void {
        var_sum += (i - mean) * (i - mean);
      });
      save_var_transform_a[f] = VarTransform<accscalar_t>{}(var_sum / n, eps);

      // update running averages
      if (running_mean.defined()) {
        running_mean_a[f] =
            momentum * mean + (1 - momentum) * running_mean_a[f];
      }
      if (running_var.defined()) {
        accscalar_t unbiased_var = var_sum / (n - 1);
        running_var_a[f] =
            momentum * unbiased_var + (1 - momentum) * running_var_a[f];
      }
    }
  });
  return std::make_tuple(save_mean, save_var_transform);
}

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
    std::array<bool, 3> grad_input_mask) {
  using accscalar_t = at::acc_type<scalar_t, false>;

  at::Tensor grad_input;
  at::Tensor grad_weight;
  at::Tensor grad_bias;
  if (grad_input_mask[0]) {
    grad_input = at::empty_like(input, input.suggest_memory_format());
  }
  if (grad_input_mask[1]) {
    grad_weight = at::empty_like(weight, at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[2]) {
    grad_bias = at::empty({input.size(1)}, weight.options());
  }

  // since we are directly manipulating pointers in contiguous path,
  // need to make sure input and grad_out have the same memory format.
  bool all_contiguous = is_contiguous(input) && is_contiguous(grad_out_) &&
      input.suggest_memory_format() == grad_out_.suggest_memory_format();

  if (all_contiguous) {
    if (grad_input_mask[0]) {
      grad_input = at::empty_like(input, suggest_memory_format_contig(input));
    }

#if defined(DYN_DISP_BUILD)
    batch_norm_cpu_backward_kernel_stub(
        kCPU,
        grad_input,
        grad_weight,
        grad_bias,
        grad_out_,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps);
#else
    batch_norm_cpu_backward_kernel_impl(
        grad_input,
        grad_weight,
        grad_bias,
        grad_out_,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps);
#endif

    return std::make_tuple(grad_input, grad_weight, grad_bias);
  }

  auto weight_a = conditional_accessor_1d<param_t>(weight);
  auto grad_weight_a = conditional_accessor_1d<param_t>(grad_weight);
  auto grad_bias_a = conditional_accessor_1d<param_t>(grad_bias);

  int64_t n_input = input.size(1);
  int64_t n = input.numel() / n_input;

  auto save_mean_a = conditional_accessor_1d<param_t>(save_mean);
  auto save_invstd_a = conditional_accessor_1d<param_t>(save_invstd);

  auto running_mean_a = conditional_accessor_1d<param_t>(running_mean);
  auto running_var_a = conditional_accessor_1d<param_t>(running_var);

  const int64_t ndim = input.dim();

  // Reduce all dimensions except dim=1
  at::DimVector reduce_dims(ndim - 1);
  reduce_dims[0] = 0;
  for (int64_t i = 2; i < ndim; ++i) {
    reduce_dims[i - 1] = i;
  }

  auto sum = at::sum(grad_out_, /*dims=*/reduce_dims);
  auto sum_a = sum.accessor<scalar_t, 1>();

  auto reduce_iter = at::TensorIteratorConfig()
                         .add_input(input)
                         .add_input(grad_out_)
                         .resize_outputs(false)
                         .declare_static_shape(input.sizes(), /*squash_dims=*/1)
                         .build();

  at::TensorIterator unary_iter;
  at::TensorIterator binary_iter;
  if (grad_input_mask[0]) {
    unary_iter.build(
        at::TensorIteratorConfig()
            .add_output(grad_input)
            .add_input(train ? input : grad_out_)
            .resize_outputs(false)
            .declare_static_shape(input.sizes(), /*squash_dims=*/1));

    if (train) {
      binary_iter.build(
          at::TensorIteratorConfig()
              .add_output(grad_input)
              .add_input(grad_input)
              .add_input(grad_out_)
              .resize_outputs(false)
              .declare_static_shape(input.sizes(), /*squash_dims=*/1));
    }
  }

  auto in_channel_stride = input.strides()[1];
  auto in_data = input.data_ptr<scalar_t>();
  auto grad_in_channel_stride =
      grad_input_mask[0] ? grad_input.strides()[1] : 0;
  auto grad_in_data =
      grad_input_mask[0] ? grad_input.data_ptr<scalar_t>() : nullptr;
  auto grad_out_channel_stride = grad_out_.strides()[1];
  auto grad_out_data = grad_out_.data_ptr<scalar_t>();

  at::parallel_for(0, n_input, 1, [&](int64_t b_begin, int64_t b_end) {
    at::TensorIterator reduce_iter_local(reduce_iter);
    at::TensorIterator unary_iter_local(unary_iter);
    at::TensorIterator binary_iter_local(binary_iter);

    for (int64_t f = b_begin; f < b_end; ++f) {
      param_t w = weight.defined() ? weight_a[f] : param_t(1);

      param_t mean, invstd;
      if (train) {
        mean = save_mean_a[f];
        invstd = save_invstd_a[f];
      } else {
        mean = running_mean_a[f];
        invstd = 1 / std::sqrt(running_var_a[f] + eps);
      }

      // dot product of the Q(X) and gradOuput
      accscalar_t dotp = 0;
      reduce_iter_local.unsafe_replace_operand(
          0, in_data + f * in_channel_stride);
      reduce_iter_local.unsafe_replace_operand(
          1, grad_out_data + f * grad_out_channel_stride);

      at::native::cpu_serial_kernel(
          reduce_iter_local, [&](const scalar_t i, const scalar_t go) -> void {
            dotp += (i - mean) * go;
          });

      if (grad_input_mask[0]) {
        if (train) {
          // when in training mode
          // Q(X) = X - E[x] ; i.e. input centered to zero mean
          // Y = Q(X) / sigma    ; i.e. BN output before weight and bias
          // dL/dX = (Q(dL/dY) - dot(Y, dL/dY) * Y) / sigma * w

          // projection of gradOutput on to output scaled by std
          scalar_t k = (scalar_t)dotp * invstd * invstd / n;
          {
            unary_iter_local.unsafe_replace_operand(
                0, grad_in_data + f * grad_in_channel_stride);
            unary_iter_local.unsafe_replace_operand(
                1, in_data + f * in_channel_stride);
            at::native::cpu_serial_kernel(
                unary_iter_local,
                [&](const scalar_t i) -> scalar_t { return (i - mean) * k; });
          }

          scalar_t grad_mean = sum_a[f] / n;
          {
            auto gI_data = grad_in_data + f * grad_in_channel_stride;
            binary_iter_local.unsafe_replace_operand(0, gI_data);
            binary_iter_local.unsafe_replace_operand(1, gI_data);
            binary_iter_local.unsafe_replace_operand(
                2, grad_out_data + f * grad_out_channel_stride);
            at::native::cpu_serial_kernel(
                binary_iter_local, [&](scalar_t gi, scalar_t go) -> scalar_t {
                  return (go - grad_mean - gi) * invstd * w;
                });
          }
        } else {
          // when in evaluation mode
          // Q(X) = X - running_mean  ; i.e. input centered to zero mean
          // Y = Q(X) / running_std    ; i.e. BN output before weight and bias
          // dL/dX = w / running_std
          {
            unary_iter_local.unsafe_replace_operand(
                0, grad_in_data + f * grad_in_channel_stride);
            unary_iter_local.unsafe_replace_operand(
                1, grad_out_data + f * grad_out_channel_stride);
            at::native::cpu_serial_kernel(
                unary_iter_local,
                [&](const scalar_t i) -> scalar_t { return i * invstd * w; });
          }
        }
      }
      if (grad_input_mask[1]) {
        grad_weight_a[f] = dotp * invstd;
      }

      if (grad_input_mask[2]) {
        grad_bias_a[f] = sum_a[f];
      }
    }
  });
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

std::tuple<at::Tensor, at::Tensor> batch_norm_update_stats_cpu(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    double momentum) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::batch_norm_update_stats_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::batch_norm_update_stats_cpu", std::vector<c10::IValue>({}));
#endif
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> running_mean_maybe_owned =
      at::borrow_from_optional_tensor(running_mean_opt);
  const at::Tensor& running_mean = *running_mean_maybe_owned;
  const at::Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return at::Tensor(); });

  const bool mixed_type = is_mixed_type(self, running_mean, running_var);
  return AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "batch_norm_update_stats_cpu",
      [&] {
        if (mixed_type) {
          checkMixedDataTypes(self, {}, {}, running_mean, running_var, {}, {});
          return batch_norm_cpu_update_stats_template<at::BFloat16, float, Var>(
              self, running_mean, running_var, momentum, 0);
        } else {
          return batch_norm_cpu_update_stats_template<scalar_t, scalar_t, Var>(
              self, running_mean, running_var, momentum, 0);
        }
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_cpu(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::batch_norm_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("torch_ipex::batch_norm_cpu", std::vector<c10::IValue>({}));
#endif
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& weight = *weight_maybe_owned;
  const at::Tensor& bias =
      c10::value_or_else(bias_opt, [] { return at::Tensor(); });
  const at::Tensor& running_mean =
      c10::value_or_else(running_mean_opt, [] { return at::Tensor(); });
  const at::Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return at::Tensor(); });

  at::checkBackend(
      "batch_norm_cpu",
      {self, weight, bias, running_mean, running_var},
      at::Backend::CPU);

  const bool mixed_type =
      is_mixed_type(self, weight, bias, running_mean, running_var);
  return AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, self.scalar_type(), "batch_norm", [&] {
        if (mixed_type) {
          checkMixedDataTypes(
              self, weight, bias, running_mean, running_var, {}, {});
          if (!train) {
            auto save_mean = at::empty({0}, self.options().dtype(at::kFloat));
            auto save_var = at::empty({0}, self.options().dtype(at::kFloat));
            return batch_norm_cpu_transform_input_template<at::BFloat16, float>(
                self,
                weight,
                bias,
                save_mean,
                save_var,
                running_mean,
                running_var,
                train,
                eps);
          } else {
            auto save_stats = batch_norm_cpu_update_stats_template<
                at::BFloat16,
                float,
                InvStd>(self, running_mean, running_var, momentum, eps);
            return batch_norm_cpu_transform_input_template<at::BFloat16, float>(
                self,
                weight,
                bias,
                std::get<0>(save_stats),
                std::get<1>(save_stats),
                running_mean,
                running_var,
                train,
                eps);
          }
        } else {
          if (!train) {
            auto save_mean = at::empty({0}, self.options());
            auto save_var = at::empty({0}, self.options());
            return batch_norm_cpu_transform_input_template<scalar_t, scalar_t>(
                self,
                weight,
                bias,
                save_mean,
                save_var,
                running_mean,
                running_var,
                train,
                eps);
          } else {
            auto save_stats = batch_norm_cpu_update_stats_template<
                scalar_t,
                scalar_t,
                InvStd>(self, running_mean, running_var, momentum, eps);
            return batch_norm_cpu_transform_input_template<scalar_t, scalar_t>(
                self,
                weight,
                bias,
                std::get<0>(save_stats),
                std::get<1>(save_stats),
                running_mean,
                running_var,
                train,
                eps);
          }
        }
      });
}

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
    std::array<bool, 3> grad_input_mask) {
#if defined(IPEX_DISP_OP)
  printf("torch_ipex::batch_norm_backward_cpu\n");
#endif
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION(
      "torch_ipex::batch_norm_backward_cpu", std::vector<c10::IValue>({}));
#endif
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& weight = *weight_maybe_owned;
  const at::Tensor& running_mean =
      c10::value_or_else(running_mean_opt, [] { return at::Tensor(); });
  const at::Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return at::Tensor(); });
  const at::Tensor& save_mean =
      c10::value_or_else(save_mean_opt, [] { return at::Tensor(); });
  const at::Tensor& save_invstd =
      c10::value_or_else(save_invstd_opt, [] { return at::Tensor(); });

  const bool mixed_type = is_mixed_type(
      self, weight, running_mean, running_var, save_mean, save_invstd);
  return AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "batch_norm_backward_cpu",
      [&] {
        if (mixed_type) {
          checkMixedDataTypes(
              self,
              weight,
              {},
              running_mean,
              running_var,
              save_mean,
              save_invstd);
          return batch_norm_backward_cpu_template<at::BFloat16, float>(
              grad_out,
              self,
              weight,
              running_mean,
              running_var,
              save_mean,
              save_invstd,
              train,
              eps,
              grad_input_mask);
        } else {
          return batch_norm_backward_cpu_template<scalar_t, scalar_t>(
              grad_out,
              self,
              weight,
              running_mean,
              running_var,
              save_mean,
              save_invstd,
              train,
              eps,
              grad_input_mask);
        }
      });
}

IPEX_TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::native_batch_norm"),
      TORCH_FN((&torch_ipex::cpu::batch_norm_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::native_batch_norm_backward"),
      TORCH_FN((&torch_ipex::cpu::batch_norm_backward_cpu)));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::batch_norm_update_stats"),
      TORCH_FN((&torch_ipex::cpu::batch_norm_update_stats_cpu)));
}

} // namespace cpu
} // namespace torch_ipex
