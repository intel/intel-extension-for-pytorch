#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include <ATen/native/Resize.h>
#include "ATen/OpMathType.h"

#include <core/Generator.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <utils/Macros.h>
#include "comm/ApplyUtils.h"

#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include <oneDNN/oneDNN.h>
#include "DistributionTemplates.h"
#include "Loops.h"
#include "LoopsTemplates.h"
#include "RandomEngine.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

template <
    typename scalar_t,
    typename accscalar_t,
    int unroll_factor,
    typename F,
    typename item_t>
inline void rrelu_with_noise_kernel(
    item_t& item,
    int numel,
    PhiloxState philox_args,
    scalar_t* output,
    scalar_t* input,
    scalar_t* noise,
    accscalar_t lower,
    accscalar_t upper,
    const F& random_func) {
  auto thread_idx = item.get_local_id(0);
  auto thread_range = item.get_local_range(0);
  auto group_idx = item.get_group(0);
  auto group_range = item.get_group_range(0);

  auto seeds = philox_unpack(philox_args);
  int idx = group_idx * thread_range + thread_idx;
  randStatePhilox4_32_10_t state;
  rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

  int range_stride = thread_range * group_range * unroll_factor;
  int rounded_size = ((numel - 1) / range_stride + 1) * range_stride;
  accscalar_t range = upper - lower;

  for (int linear_index = idx; linear_index < rounded_size;
       linear_index += range_stride) {
    auto rand = random_func(&state);

    // ensure that (&rand.x)[ii] is safe
    static_assert(sizeof(rand) / sizeof(rand.x) == unroll_factor, "");

#pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + thread_range * group_range * ii;
      if (li >= numel) {
        continue;
      }
      scalar_t r = static_cast<scalar_t>((&rand.x)[ii]);
      r = r * range + lower;
      if (input[li] <= 0) {
        output[li] = input[li] * r;
        noise[li] = r;
      } else {
        output[li] = input[li];
        noise[li] = static_cast<scalar_t>(1);
      }
    }
  }
}

template <typename scalar_t>
inline void _rrelu_with_noise_train(
    Tensor& output,
    const Tensor& input_,
    const Tensor& noise_,
    const Scalar& lower_,
    const Scalar& upper_,
    c10::optional<Generator> generator) {
  auto& sycl_queue = dpcppGetCurrentQueue();
  auto input = input_.contiguous();
  auto noise = noise_.contiguous();
  Tensor tmp_output = output.contiguous();

  int64_t numel = input.numel();
  auto execution_policy = calc_execution_policy(numel);

  auto counter_offset = std::get<0>(execution_policy);
  auto num_groups = std::get<1>(execution_policy);
  auto group_size = std::get<2>(execution_policy);

  auto gen = at::get_generator_or_default<DPCPPGeneratorImpl>(
      generator, getDefaultDPCPPGenerator());
  std::pair<uint64_t, uint64_t> seeds;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    seeds = gen->philox_engine_inputs(counter_offset);
  }
  PhiloxState rng_engine_inputs(std::get<0>(seeds), std::get<1>(seeds));

  scalar_t* input_data = input.data_ptr<scalar_t>();
  scalar_t* noise_data = noise.data_ptr<scalar_t>();
  scalar_t* output_data = tmp_output.data_ptr<scalar_t>();

  double lower = lower_.to<double>();
  double upper = upper_.to<double>();

  if (std::is_same<scalar_t, double>::value) {
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        rrelu_with_noise_kernel<scalar_t, double, 2>(
            item,
            numel,
            rng_engine_inputs,
            output_data,
            input_data,
            noise_data,
            lower,
            upper,
            [](randStatePhilox4_32_10_t* state) {
              return rand_uniform2_double(state);
            });
      };
      cgh.parallel_for(
          sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  } else {
    auto lower_ = static_cast<float>(lower);
    auto upper_ = static_cast<float>(upper);
    // half and float
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
        rrelu_with_noise_kernel<scalar_t, float, 4>(
            item,
            numel,
            rng_engine_inputs,
            output_data,
            input_data,
            noise_data,
            lower_,
            upper_,
            [](randStatePhilox4_32_10_t* state) {
              return rand_uniform4(state);
            });
      };
      cgh.parallel_for(
          sycl::nd_range<1>(num_groups * group_size, group_size), kfn);
    };
    DPCPP_Q_SUBMIT(sycl_queue, cgf);
  }

  if (!output.is_contiguous()) {
    output.copy_(tmp_output);
  }
}

inline void _prelu_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "prelu_dpcpp", [&] {
        dpcpp_kernel_for_tensor_iter(
            iter, [=](scalar_t input, scalar_t weight) -> scalar_t {
              return (input > 0) ? input : weight * input;
            });
      });
}

inline void _prelu_kernel_backward_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "prelu_backward_dpcpp", [&] {
        dpcpp_kernel_multiple_outputs_for_tensor_iter(
            iter,
            [=](scalar_t input,
                scalar_t weight,
                scalar_t grad) -> std::tuple<scalar_t, scalar_t> {
              auto mask = input > 0;
              auto grad_input = mask ? grad : weight * grad;
              auto grad_weight = mask ? scalar_t{0} : input * grad;
              return {grad_input, grad_weight};
            });
      });
}

inline Tensor threshold_out(
    optional<Tensor> opt_result,
    const Tensor& self,
    const Scalar& threshold,
    const Scalar& value,
    const Tensor& other) {
  Tensor result = opt_result.value_or(Tensor());
  return unary_out_with_onednn_and_loops_bw<dnnl::algorithm::eltwise_relu>(
      TensorIterator::binary_op,
      result,
      self,
      other,
      [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            iter.dtype(),
            "threshold",
            [&] {
              scalar_t _threshold = threshold.to<scalar_t>();
              scalar_t _value = value.to<scalar_t>();
              dpcpp_kernel_for_tensor_iter(
                  iter, [=](scalar_t x, scalar_t other) -> scalar_t {
                    return x <= _threshold ? _value : other;
                  });
            });
      },
      0.0f,
      0.0f,
      0.0 == threshold.to<float>() && 0.0 == value.to<float>());
}

template <typename scalar_t>
inline scalar_t relu_forward(scalar_t self) {
  return self > 0 ? self : static_cast<scalar_t>(0);
}

template <typename scalar_t>
inline scalar_t gelu_erf_forward(scalar_t x) {
  using opmath_t = at::opmath_type<scalar_t>;
  constexpr opmath_t kAlpha = M_SQRT1_2;
  return static_cast<opmath_t>(x) * opmath_t(0.5) *
      (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));
}

template <typename scalar_t>
inline scalar_t gelu_tanh_forward(scalar_t x) {
  using opmath_t = at::opmath_type<scalar_t>;
  constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * opmath_t(0.5);
  constexpr opmath_t kKappa = 0.044715;
  auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x) *
      static_cast<opmath_t>(x);
  auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
  return opmath_t(0.5) * static_cast<opmath_t>(x) *
      (opmath_t(1) + Numerics<opmath_t>::tanh(inner));
}

template <typename scalar_t>
inline scalar_t gelu_erf_backward(scalar_t dy, scalar_t x) {
  using opmath_t = at::opmath_type<scalar_t>;
  constexpr opmath_t kBeta = M_2_SQRTPI * M_SQRT1_2 * opmath_t(0.5);
  constexpr opmath_t kAlpha = M_SQRT1_2;
  const opmath_t cdf =
      opmath_t(0.5) * (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));
  const opmath_t pdf = Numerics<opmath_t>::exp(
                           opmath_t(-0.5) * static_cast<opmath_t>(x) *
                           static_cast<opmath_t>(x)) *
      kBeta;
  return static_cast<opmath_t>(dy) * (cdf + static_cast<opmath_t>(x) * pdf);
}

template <typename scalar_t>
inline scalar_t gelu_tanh_backward(scalar_t dy, scalar_t x) {
  using opmath_t = at::opmath_type<scalar_t>;
  constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * opmath_t(0.5);
  constexpr opmath_t kKappa = 0.044715;
  auto x_sq = static_cast<opmath_t>(x) * static_cast<opmath_t>(x);
  auto x_cube = x_sq * static_cast<opmath_t>(x);
  auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
  auto tanh_inner = Numerics<opmath_t>::tanh(inner);

  auto left = opmath_t(0.5) * static_cast<opmath_t>(x);
  auto right = opmath_t(1) + tanh_inner;

  auto left_derivative = 0.5 * right;

  auto tanh_derivative = opmath_t(1) - tanh_inner * tanh_inner;
  auto inner_derivative = kBeta * (opmath_t(1) + opmath_t(3) * kKappa * x_sq);
  auto right_derivative = left * tanh_derivative * inner_derivative;

  return static_cast<opmath_t>(dy) * (left_derivative + right_derivative);
}

Tensor& silu_out_kernel(const Tensor& self, Tensor& result) {
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_swish>(
      TensorIterator::unary_op,
      result,
      self,
      [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            iter.dtype(),
            "_silu_out",
            [&]() {
              dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t x) -> scalar_t {
                using accscalar_t = acc_type<scalar_t>;
                const accscalar_t one = 1.0f;
                return static_cast<accscalar_t>(x) /
                    (one +
                     Numerics<accscalar_t>::exp(-static_cast<accscalar_t>(x)));
              });
            });
      },
      /* alpha = */ 1.0f);
}

} // namespace impl

Tensor relu(const Tensor& self) {
  Tensor result;
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_relu>(
      TensorIterator::unary_op, result, self, [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            iter.dtype(),
            "relu",
            [&]() {
              dpcpp_kernel_for_tensor_iter(
                  iter, [=](scalar_t self) -> scalar_t {
                    return impl::relu_forward<scalar_t>(self);
                  });
            });
      });
}

Tensor& relu_(Tensor& self) {
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_relu>(
      TensorIterator::unary_op, self, self, [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            iter.dtype(),
            "relu_",
            [&]() {
              dpcpp_kernel_for_tensor_iter(
                  iter, [=](scalar_t self) -> scalar_t {
                    return impl::relu_forward<scalar_t>(self);
                  });
            });
      });
}

Tensor& threshold_(Tensor& self, const Scalar& threshold, const Scalar& value) {
  impl::threshold_out(make_optional(self), self, threshold, value, self);
  return self;
}

Tensor threshold(
    const Tensor& self,
    const Scalar& threshold,
    const Scalar& value) {
  return impl::threshold_out(nullopt, self, threshold, value, self);
}

Tensor& threshold_out(
    const Tensor& self,
    const Scalar& threshold,
    const Scalar& value,
    Tensor& result) {
  impl::threshold_out(make_optional(result), self, threshold, value, self);
  return result;
}

Tensor threshold_backward(
    const Tensor& grad,
    const Tensor& self,
    const Scalar& threshold) {
  return impl::threshold_out(nullopt, self, threshold, 0, grad);
}

Tensor& threshold_backward_out(
    const Tensor& grad,
    const Tensor& self,
    const Scalar& threshold,
    Tensor& gradInput) {
  impl::threshold_out(make_optional(gradInput), self, threshold, 0, grad);
  return gradInput;
}

Tensor& rrelu_with_noise_out(
    const Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    c10::optional<Generator> generator,
    Tensor& out) {
  at::native::resize_output(out, self.sizes());
  if (self.numel() == 0) {
    return out;
  }
  if (training) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "rrelu_with_noise_out",
        [&] {
          impl::_rrelu_with_noise_train<scalar_t>(
              out, self, noise, lower, upper, generator);
        });
  } else {
    auto lower_tensor = lower.to<double>();
    auto upper_tensor = upper.to<double>();
    Scalar negative_slope = (lower_tensor + upper_tensor) / 2;
    at::leaky_relu_out(out, self, negative_slope);
  }
  return out;
}

Tensor rrelu_with_noise(
    const Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    c10::optional<Generator> generator) {
  Tensor output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return rrelu_with_noise_out(
      self, noise, lower, upper, training, generator, output);
}

Tensor& rrelu_with_noise_(
    Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    c10::optional<Generator> generator) {
  return rrelu_with_noise_out(
      self, noise, lower, upper, training, generator, self);
}

Tensor rrelu_with_noise_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    bool self_is_result) {
  if (training) {
    return noise * grad_output;
  } else {
    auto l = lower.toDouble();
    auto u = upper.toDouble();
    auto mid = (l + u) / 2.;
    return at::leaky_relu_backward(grad_output, self, mid, self_is_result);
  }
}

Tensor _prelu_kernel(const Tensor& self, const Tensor& weight) {
  // Weight broadcasts over self and they have the same dtype
  auto result = at::empty_like(self);
  auto iter = TensorIteratorConfig()
                  .add_output(result)
                  .add_input(self)
                  .add_input(weight)
                  .build();
  impl::_prelu_kernel_dpcpp(iter);
  return result;
}

std::tuple<Tensor, Tensor> _prelu_kernel_backward(
    const Tensor& grad_out,
    const Tensor& self,
    const Tensor& weight) {
  Tensor grad_self = at::empty({0}, self.options());
  Tensor grad_weight = at::empty({0}, weight.options());
  auto iter = TensorIteratorConfig()
                  .add_output(grad_self)
                  .add_output(grad_weight)
                  .add_input(self)
                  .add_input(weight)
                  .add_input(grad_out)
                  .build();
  impl::_prelu_kernel_backward_dpcpp(iter);
  return {grad_self, grad_weight};
}

Tensor& hardshrink_out(
    const Tensor& self,
    const Scalar& lambd,
    Tensor& result) {
  auto iter = TensorIterator::unary_op(result, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "hardshrink",
      [&]() {
        auto _lambd = lambd.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t x) -> scalar_t {
          return (x >= -_lambd && x <= _lambd) ? scalar_t(0) : x;
        });
      });
  return result;
}

Tensor hardshrink(const Tensor& self, const Scalar& lambd) {
  Tensor result = at::empty_like(self);
  return hardshrink_out(self, lambd, result);
}

Tensor& hardshrink_backward_out(
    const Tensor& grad,
    const Tensor& self,
    const Scalar& lambd,
    Tensor& grad_input) {
  auto iter = TensorIterator::binary_op(grad_input, grad, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "hardshrink_backward_out",
      [&]() {
        auto _lambd = lambd.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(
            iter, [=](scalar_t grad_output, scalar_t x) -> scalar_t {
              return (x >= -_lambd && x <= _lambd) ? scalar_t(0) : grad_output;
            });
      });
  return grad_input;
}

Tensor hardshrink_backward(
    const Tensor& grad,
    const Tensor& self,
    const Scalar& lambd) {
  auto result = at::empty_like(grad);
  return hardshrink_backward_out(grad, self, lambd, result);
}

Tensor& hardswish_out(const Tensor& self, Tensor& result) {
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_hardswish>(
      TensorIterator::unary_op,
      result,
      self,
      [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            iter.dtype(),
            "hardswish",
            [&]() {
              using accscalar_t = acc_type<scalar_t>;
              const accscalar_t zero(0.0f);
              const accscalar_t one_sixth(1.0f / 6.0f);
              const accscalar_t three(3.0f);
              const accscalar_t six(6.0f);
              dpcpp_kernel_for_tensor_iter(
                  iter,
                  [zero, one_sixth, three, six](scalar_t self_val) -> scalar_t {
                    accscalar_t x = static_cast<accscalar_t>(self_val);
                    return x *
                        Numerics<accscalar_t>::min(
                               Numerics<accscalar_t>::max(x + three, zero),
                               six) *
                        one_sixth;
                  });
            });
      },
      /* alpha = */ 1.0f / 6.0f,
      /* beta = */ 1.0f / 2.0f);
}

Tensor hardswish(const Tensor& self) {
  Tensor result = at::empty_like(self);
  return at::AtenIpexTypeXPU::hardswish_out(self, result);
}

Tensor& hardswish_(Tensor& self) {
  return at::AtenIpexTypeXPU::hardswish_out(self, self);
}

Tensor hardswish_backward(const Tensor& grad_output, const Tensor& self) {
  auto result = at::empty_like(grad_output);
  auto iter = TensorIterator::binary_op(result, grad_output, self);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "hardswish_backward",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        const accscalar_t zero(0.0f);
        const accscalar_t three(3.0f);
        const accscalar_t neg_three(-3.0f);
        const accscalar_t one_half(0.5f);
        dpcpp_kernel_for_tensor_iter(
            iter,
            [zero, three, neg_three, one_half](
                scalar_t grad_val_, scalar_t self_val_) -> scalar_t {
              accscalar_t grad_val = static_cast<accscalar_t>(grad_val_);
              accscalar_t self_val = static_cast<accscalar_t>(self_val_);
              if (self_val < neg_three) {
                return zero;
              } else if (self_val <= three) {
                return grad_val * ((self_val / three) + one_half);
              } else {
                return grad_val;
              }
            });
      });
  return result;
}

Tensor& gelu_out(
    const Tensor& self,
    c10::string_view approximate,
    Tensor& result) {
  auto _approximate = at::native::get_gelutype_enum(approximate);
  if (_approximate == at::native::GeluType::Tanh) {
    return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_gelu_tanh>(
        TensorIterator::unary_op, result, self, [=](TensorIteratorBase& iter) {
          IPEX_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::BFloat16,
              at::ScalarType::Half,
              iter.dtype(),
              "gelu",
              [&]() {
                dpcpp_kernel_for_tensor_iter(
                    iter, [=](scalar_t self) -> scalar_t {
                      return impl::gelu_tanh_forward<scalar_t>(self);
                    });
              });
        });
  } else {
    return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_gelu_erf>(
        TensorIterator::unary_op, result, self, [=](TensorIteratorBase& iter) {
          IPEX_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::BFloat16,
              at::ScalarType::Half,
              iter.dtype(),
              "gelu",
              [&]() {
                dpcpp_kernel_for_tensor_iter(
                    iter, [=](scalar_t self) -> scalar_t {
                      return impl::gelu_erf_forward<scalar_t>(self);
                    });
              });
        });
  }
}

Tensor gelu(const Tensor& self, c10::string_view approximate) {
  Tensor result;
  return gelu_out(self, approximate, result);
}

Tensor& gelu_backward_out(
    const Tensor& grad,
    const Tensor& self,
    c10::string_view approximate,
    Tensor& grad_input) {
  auto _approximate = at::native::get_gelutype_enum(approximate);
  if (_approximate == at::native::GeluType::Tanh) {
    return unary_out_with_onednn_and_loops_bw<
        dnnl::algorithm::eltwise_gelu_tanh>(
        TensorIterator::binary_op,
        grad_input,
        self,
        grad,
        [=](TensorIteratorBase& iter) {
          IPEX_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::BFloat16,
              at::ScalarType::Half,
              iter.dtype(),
              "gelu_backward",
              [&]() {
                dpcpp_kernel_with_scalars(
                    iter, [=](scalar_t self, scalar_t grad) -> scalar_t {
                      return impl::gelu_tanh_backward<scalar_t>(grad, self);
                    });
              });
        });
  } else {
    return unary_out_with_onednn_and_loops_bw<
        dnnl::algorithm::eltwise_gelu_erf>(
        TensorIterator::binary_op,
        grad_input,
        self,
        grad,
        [=](TensorIteratorBase& iter) {
          IPEX_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::BFloat16,
              at::ScalarType::Half,
              iter.dtype(),
              "gelu_backward",
              [&]() {
                dpcpp_kernel_with_scalars(
                    iter, [=](scalar_t self, scalar_t grad) -> scalar_t {
                      return impl::gelu_erf_backward<scalar_t>(grad, self);
                    });
              });
        });
  }
}

Tensor gelu_backward(
    const Tensor& grad,
    const Tensor& self,
    c10::string_view approximate) {
  Tensor result;
  return gelu_backward_out(grad, self, approximate, result);
}

Tensor& silu_out(const Tensor& self, Tensor& output) {
  return impl::silu_out_kernel(self, output);
}

Tensor& silu_backward_out(
    const Tensor& grad_output,
    const Tensor& output,
    Tensor& grad_input) {
  return unary_out_with_onednn_and_loops_bw<dnnl::algorithm::eltwise_swish>(
      TensorIterator::binary_op,
      grad_input,
      grad_output,
      output,
      [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            iter.dtype(),
            "silu_backward_out",
            [&]() {
              dpcpp_kernel_for_tensor_iter(
                  iter, [=](scalar_t dy, scalar_t x) -> scalar_t {
                    using accscalar_t = acc_type<scalar_t>;
                    const accscalar_t dy_acc = static_cast<accscalar_t>(dy);
                    const accscalar_t x_acc = static_cast<accscalar_t>(x);
                    const accscalar_t one = 1.0f;
                    const accscalar_t s_acc =
                        one / (one + Numerics<accscalar_t>::exp(-x_acc));
                    return dy_acc * s_acc * (one + x_acc * (one - s_acc));
                  });
            });
      },
      /* alpha = */ 1.0f);
}

Tensor silu_backward(const Tensor& grad_output, const Tensor& output) {
  Tensor grad_input = at::empty_like(grad_output);
  return silu_backward_out(grad_output, output, grad_input);
}

at::Tensor& mish_out(const at::Tensor& self, at::Tensor& out) {
  return unary_out_with_onednn_and_loops<dnnl::algorithm::eltwise_mish>(
      TensorIterator::unary_op, out, self, [=](TensorIteratorBase& iter) {
        IPEX_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            iter.dtype(),
            "mish",
            [&]() {
              dpcpp_kernel_for_tensor_iter(
                  iter, [=](scalar_t self) -> scalar_t {
                    using accscalar_t = acc_type<scalar_t>;
                    const accscalar_t x_acc = static_cast<accscalar_t>(self);
                    return x_acc *
                        Numerics<accscalar_t>::tanh(
                               Numerics<accscalar_t>::log1p(
                                   Numerics<accscalar_t>::exp(x_acc)));
                  });
            });
      });
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("silu_backward", TORCH_FN(silu_backward));
}

} // namespace AtenIpexTypeXPU
} // namespace at
