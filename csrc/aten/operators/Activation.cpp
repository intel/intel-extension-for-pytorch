#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>
#include "ATen/OpMathType.h"

#include <core/Generator.h>
#include <core/TensorImplUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <utils/Macros.h>
#include "comm/ApplyUtils.h"

#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include <oneDNN/oneDNN.h>
#include "Loops.h"
#include "Random.h"

#include <iostream>
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

namespace impl {

template <typename scalar_t>
static inline bool is_contiguous(const int64_t* strides) {
  return strides[0] == sizeof(scalar_t) && strides[1] == sizeof(scalar_t) &&
      strides[2] == sizeof(scalar_t);
}

template <typename scalar_t>
static void RReLU_updateOutput(
    const Tensor& input,
    Tensor& output,
    const Tensor& noise,
    double lower,
    double upper,
    bool train,
    bool inplace,
    c10::optional<Generator> generator) {
  auto gen = at::get_generator_or_default<DPCPPGeneratorImpl>(
      generator, getDefaultDPCPPGenerator());
  if (train) {
    auto input_ = input.contiguous();
    noise.resize_as_(input_);

    std::pair<uint64_t, uint64_t> seeds;
    {
      // See Note [Acquire lock when using random generators]
      // this philox_engine_inputs('1') is aligned with Distribution.cpp,
      // yet they use '((n - 1) / (BLOCK_SIZE * grid.x) + 1) *
      // curand4_engine_calls' in the same place.
      std::lock_guard<std::mutex> lock(gen->mutex_);
      seeds = gen->philox_engine_inputs(1);
    }
    if (inplace) {
      auto& dpcpp_queue = dpcppGetCurrentQueue();
      auto total_threads = input_.numel();

      auto cgf = DPCPP_Q_CGF(cgh) {
        auto in_data = input_.data_ptr<scalar_t>();
        auto noise_data = noise.data_ptr<scalar_t>();
        cgh.parallel_for(
            sycl::range<1>(total_threads), [=](sycl::item<1> itemId) {
              auto in_ptr = in_data;
              auto noise_ptr = noise_data;
              auto id = itemId.get_id(0);
              auto linear_id = itemId.get_linear_id();

              RandomState<Philox4_32_10> state(
                  seeds.first, linear_id, seeds.second);

              if (in_ptr[id] <= 0) {
                double rand = state.uniform<double>();
                scalar_t r = ScalarConvert<double, scalar_t>::to(
                    rand * (upper - lower) + lower);
                in_ptr[id] = static_cast<scalar_t>(in_ptr[id]) * r;
                noise_ptr[id] = r;
              } else {
                noise_ptr[id] = 1;
              }
            });
      };
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
      output.set_(input_);
    } else {
      output.resize_as_(input_);

      auto& dpcpp_queue = dpcppGetCurrentQueue();
      auto total_threads = input_.numel();

      auto cgf = DPCPP_Q_CGF(cgh) {
        auto in_data = input_.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        auto noise_data = noise.data_ptr<scalar_t>();
        cgh.parallel_for(
            sycl::range<1>(total_threads), [=](sycl::item<1> itemId) {
              auto in_ptr = in_data;
              auto out_ptr = out_data;
              auto noise_ptr = noise_data;
              auto id = itemId.get_id(0);
              auto linear_id = itemId.get_linear_id();

              RandomState<Philox4_32_10> state(
                  seeds.first, linear_id, seeds.second);

              if (in_ptr[id] <= 0) {
                double rand = state.uniform<double>();
                scalar_t r = ScalarConvert<double, scalar_t>::to(
                    rand * (upper - lower) + lower);
                out_ptr[id] = static_cast<scalar_t>(in_ptr[id]) * r;
                noise_ptr[id] = r;
              } else {
                out_ptr[id] = in_ptr[id];
                noise_ptr[id] = 1;
              }
            });
      };
      DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
    }
  } else {
    const scalar_t negSlope =
        ScalarConvert<double, scalar_t>::to((lower + upper) / 2);
    output.resize_as_(input);
    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(true)
                    .add_output(output)
                    .add_input(input)
                    .build();
    dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t in) -> scalar_t {
      return (in <= 0) ? in * negSlope : in;
    });
  }
}

template <typename scalar_t>
static void RReLU_updateGradInput(
    const Tensor& input,
    const Tensor& gradOutput,
    Tensor& gradInput,
    const Tensor& noise,
    double lower,
    double upper,
    bool train,
    bool inplace) {
  TORCH_CHECK(
      input.numel() == gradOutput.numel(),
      "input and gradOutput have different number of elements");
  if (train && upper - lower > 1E-6) {
    if (inplace) {
      gradOutput.mul_(noise);
      gradInput.set_(gradOutput);
    } else {
      gradInput.resize_as_(input);
      at::AtenIpexTypeXPU::mul_out(gradOutput, noise, gradInput);
    }
  } else {
    const scalar_t negSlope =
        ScalarConvert<double, scalar_t>::to((lower + upper) / 2);
    gradInput.resize_as_(input);
    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(true)
                    .add_output(gradInput)
                    .add_input(gradOutput)
                    .add_input(input)
                    .build();
    dpcpp_kernel_for_tensor_iter(
        iter, [=](scalar_t grad_out, scalar_t in) -> scalar_t {
          return (in <= 0) ? grad_out * negSlope : grad_out;
        });
  }
}

template <typename scalar_t>
void inline prelu_backward_kernel_share_weights(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& grad_out,
    Tensor& input_grad,
    Tensor& weight_grad_collector) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = input_grad.numel();
  auto weight_val = weight.data_ptr<scalar_t>();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_grad_data = input_grad.data_ptr<scalar_t>();
    auto weight_grad_collector_data =
        weight_grad_collector.data_ptr<scalar_t>();
    auto in_data = input.data_ptr<scalar_t>();
    auto grad_out_data = grad_out.data_ptr<scalar_t>();

    cgh.parallel_for(sycl::range<1>(total_threads), [=](sycl::item<1> itemId) {
      auto in_grad_ptr = in_grad_data;
      auto weight_grad_collector_ptr = weight_grad_collector_data;
      auto in_ptr = in_data;
      auto grad_out_ptr = grad_out_data;
      auto id = itemId.get_id(0);

      in_grad_ptr[id] = (in_ptr[id] > 0)
          ? grad_out_ptr[id]
          : (*weight_val) * static_cast<scalar_t>(grad_out_ptr[id]);
      weight_grad_collector_ptr[id] = (in_ptr[id] > 0)
          ? scalar_t(0)
          : static_cast<scalar_t>(in_ptr[id]) *
              static_cast<scalar_t>(grad_out_ptr[id]);
    });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t>
void inline prelu_backward_kernel_multi_weights(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& grad_out,
    Tensor& input_grad,
    Tensor& weight_grad_collector,
    int64_t input_dim0_size,
    int64_t channel_size,
    int64_t input_stride0,
    int64_t input_stride1) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = input.numel();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = input.data_ptr<scalar_t>();
    auto weight_data = weight.data_ptr<scalar_t>();
    auto gred_out_data = grad_out.data_ptr<scalar_t>();
    auto in_grad_data = input_grad.data_ptr<scalar_t>();
    auto weight_grad_collector_data =
        weight_grad_collector.data_ptr<scalar_t>();
    cgh.parallel_for(sycl::range<1>(total_threads), [=](sycl::item<1> itemId) {
      auto in_ptr = in_data;
      auto weight_ptr = weight_data;
      auto grad_out_ptr = gred_out_data;
      auto in_grad_ptr = in_grad_data;
      auto weight_grad_collector_ptr = weight_grad_collector_data;
      auto id = itemId.get_id(0);

      int64_t channel = (id % input_stride0) / input_stride1;
      scalar_t input_data_val = in_ptr[id];
      scalar_t grad_out_data_val = grad_out_ptr[id];
      in_grad_ptr[id] = (input_data_val > 0)
          ? grad_out_data_val
          : static_cast<scalar_t>(weight_ptr[channel]) * grad_out_data_val;
      weight_grad_collector_ptr[id] = (input_data_val > 0)
          ? scalar_t(0)
          : input_data_val * grad_out_data_val;
    });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

inline Tensor threshold_out(
    optional<Tensor> opt_result,
    const Tensor& self,
    const Scalar& threshold,
    const Scalar& value,
    const Tensor& other) {
  Tensor result = opt_result.value_or(Tensor());
  if (IPEX_ANY(xpu::oneDNN::is_onednn_layout, self, other) &&
      0.0 == threshold.to<float>() && 0.0 == value.to<float>() &&
      IPEX_ALL(xpu::oneDNN::eltwise_backward_valid, self, other)) {
    // need or not
    // 1. input is oneDNN layout
    // 2. it is a relu bwd (threshold and value)
    // can or not
    // 1. input is a valid memory supported by oneDNN
    xpu::oneDNN::eltwise_backward<dnnl::algorithm::eltwise_relu>(
        result, self, other, 0.0f, 0.0f);
    return result;
  } else {
    auto _self = to_plain_if_needed(self);
    auto _other = to_plain_if_needed(other);
    auto iter = TensorIterator::binary_op(result, _self, _other);
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
    return iter.output();
  }
}

template <typename scalar_t>
inline scalar_t relu_forward(scalar_t self, scalar_t alpha) {
  return self >= 0 ? self : self * alpha;
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
inline scalar_t gelu_erf_backward(scalar_t grad, scalar_t self) {
  using accscalar_t = acc_type<scalar_t>;
  auto v = static_cast<accscalar_t>(self) * M_SQRT1_2;
  return (scalar_t)(
      grad * 0.5 *
      (1.0 + Numerics<accscalar_t>::erf(v) +
       v * M_2_SQRTPI * Numerics<accscalar_t>::exp(-v * v)));
}

Tensor& silu_out_kernel(const Tensor& self, Tensor& result) {
  auto iter = TensorIterator::unary_float_op(result, self);
  IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "_silu_out",
      [&]() {
        result.resize_as_(self);
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t x) -> scalar_t {
          scalar_t one = (scalar_t)1.0;
          return x / (one + Numerics<scalar_t>::exp(-static_cast<scalar_t>(x)));
        });
      });
  return result;
}

template <typename scalar_t>
void silu_backward_kernel(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& self) {
  gradInput.resize_as_(self);
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(gradInput)
                  .add_input(gradOutput)
                  .add_input(self)
                  .build();
  dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t dy, scalar_t x) -> scalar_t {
    scalar_t one = (scalar_t)1.0;
    const scalar_t sigmoid =
        one / (one + Numerics<scalar_t>::exp(-static_cast<scalar_t>(x)));
    return dy * sigmoid * (one + x * (one - sigmoid));
  });
}

template <typename scalar_t>
inline scalar_t mish_forward(scalar_t self) {
  using T_ACC = acc_type<scalar_t>;
  const T_ACC x_acc = static_cast<T_ACC>(self);
  return (scalar_t)(
      x_acc *
      Numerics<T_ACC>::tanh(
          Numerics<T_ACC>::log1p(Numerics<T_ACC>::exp(x_acc))));
}
} // namespace impl

Tensor relu(const Tensor& self) {
  if (xpu::oneDNN::is_onednn_layout(self) &&
      xpu::oneDNN::eltwise_forward_valid(self)) {
    Tensor result;
    xpu::oneDNN::eltwise<dnnl::algorithm::eltwise_relu>(
        result, self, 0.0f, 0.0f);
    return result;
  } else {
    auto _self = to_plain_if_needed(self);
    auto result = at::empty_like(_self);
    auto iter = TensorIterator::unary_op(result, _self);
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        iter.dtype(),
        "relu",
        [&]() {
          dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t self) -> scalar_t {
            return impl::relu_forward<scalar_t>(self);
          });
        });
    return result;
  }
}

Tensor& relu_(Tensor& self) {
  if (xpu::oneDNN::is_onednn_layout(self) &&
      xpu::oneDNN::eltwise_forward_valid(self)) {
    xpu::oneDNN::eltwise<dnnl::algorithm::eltwise_relu>(self, self, 0.0f, 0.0f);
    return self;
  } else {
    self = to_plain_if_needed_(self);
    auto iter = TensorIterator::unary_op(self, self);
    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        iter.dtype(),
        "relu_",
        [&]() {
          dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t self) -> scalar_t {
            return impl::relu_forward<scalar_t>(self);
          });
        });
    return self;
  }
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

Tensor rrelu_with_noise(
    const Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    c10::optional<Generator> generator) {
  auto self_ = self.contiguous();
  Tensor output = at::empty_like(self_);
  auto lower_ = lower.toDouble();
  auto upper_ = upper.toDouble();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "RReLU_updateOutput",
      [&]() {
        impl::RReLU_updateOutput<scalar_t>(
            self, output, noise, lower_, upper_, training, false, generator);
      });
  return output;
}
// TODO: fix const self
Tensor& rrelu_with_noise_(
    Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    c10::optional<Generator> generator) {
  auto lower_ = lower.toDouble();
  auto upper_ = upper.toDouble();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "RReLU_updateOutput",
      [&]() {
        impl::RReLU_updateOutput<scalar_t>(
            self, self, noise, lower_, upper_, training, true, generator);
      });
  return self;
}

Tensor& rrelu_with_noise_out(
    const Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    c10::optional<Generator> generator,
    Tensor& out) {
  auto lower_ = lower.toDouble();
  auto upper_ = upper.toDouble();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "RReLU_updateOutput",
      [&]() {
        impl::RReLU_updateOutput<scalar_t>(
            self, out, noise, lower_, upper_, training, false, generator);
      });
  return out;
}

Tensor rrelu_with_noise_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    bool self_is_result) {
  TORCH_CHECK(
      !self_is_result,
      "In-place rrelu_ backward calculation is triggered with a negative slope which is not supported. "
      "This is caused by calling in-place forward function with a negative slope, "
      "please call out-of-place version instead.");
  Tensor grad_input = at::empty_like(grad_output);
  auto lower_ = lower.toDouble();
  auto upper_ = upper.toDouble();
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "RReLU_updateGradInput",
      [&]() {
        impl::RReLU_updateGradInput<scalar_t>(
            grad_output,
            self,
            grad_input,
            noise,
            lower_,
            upper_,
            training,
            self_is_result);
      });
  return grad_input;
}

Tensor prelu(const Tensor& self, const Tensor& weight) {
  auto result = at::empty_like(self);
  auto iter = TensorIterator::binary_op(result, self, weight);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "prelu",
      [&]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [=](scalar_t x, scalar_t w) -> scalar_t {
              return impl::relu_forward<scalar_t>(x, w);
            });
      });
  return result;
}

std::tuple<Tensor, Tensor> prelu_backward(
    const Tensor& grad_out_,
    const Tensor& self,
    const Tensor& weight_) {
  auto input = self.contiguous();
  auto grad_out = grad_out_.contiguous();
  auto weight = weight_.contiguous();

  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(grad_out.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());

  int64_t weight_num = weight.numel();
  auto strides = input.strides();
  auto dims = input.dim();

  Tensor input_grad = at::empty_like(input);
  Tensor weight_grad = at::empty_like(weight);
  Tensor weight_grad_collector = at::empty_like(input);

  if (weight_num == 1) {
    IPEX_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16, input.scalar_type(), "prelu_backward", [&] {
          impl::prelu_backward_kernel_share_weights<scalar_t>(
              input, weight, grad_out, input_grad, weight_grad_collector);
        });
    // fix me: fill_() returns RuntimeError when input
    // weight_grad_collector.sum() is without '.item()'
    weight_grad.fill_(weight_grad_collector.sum().item());
  } else {
    int64_t input_ndim = input.dim();
    TORCH_CHECK(input_ndim > 0, "Not allow zero-dim input tensor.");

    int64_t channel_size = 1;
    int64_t input_dim0_size = 1, input_stride0 = 1, input_stride1 = 1;

    if (input_ndim > 1) {
      channel_size = input.size(1);
      input_dim0_size = input.size(0);
      input_stride0 = strides[0];
      input_stride1 = strides[1];
    }
    TORCH_CHECK(
        channel_size == weight_num,
        "Mismatch of parameter numbers and input channel size. Found parameter numbers = ",
        weight_num,
        " and channel size = ",
        channel_size,
        ".");

    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "prelu_backward",
        [&] {
          impl::prelu_backward_kernel_multi_weights<scalar_t>(
              input,
              weight,
              grad_out,
              input_grad,
              weight_grad_collector,
              input_dim0_size,
              channel_size,
              input_stride0,
              input_stride1);
        });
    // update weight_grad

    std::vector<int64_t> reduce_dims;
    reduce_dims.push_back(0);
    if (dims > 2) {
      for (int64_t i = 2; i < dims; i++)
        reduce_dims.push_back(i);
    }
    weight_grad = weight_grad_collector.sum(reduce_dims);
  }
  return std::tuple<Tensor, Tensor>{input_grad, weight_grad};
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
  auto iter = TensorIterator::unary_op(result, self);
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
            iter, [zero, one_sixth, three, six](scalar_t self_val) -> scalar_t {
              accscalar_t x = static_cast<accscalar_t>(self_val);
              return x *
                  Numerics<accscalar_t>::min(
                         Numerics<accscalar_t>::max(x + three, zero), six) *
                  one_sixth;
            });
      });
  return result;
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
  at::native::GeluType approximate_gelu =
      at::native::get_gelutype_enum(approximate);
  if (xpu::oneDNN::is_onednn_layout(self) &&
      xpu::oneDNN::eltwise_forward_valid(self)) {
    if (approximate_gelu == at::native::GeluType::Tanh) {
      std::cout << "--------onednn go to tanh path------" << std::endl;
      xpu::oneDNN::eltwise<dnnl::algorithm::eltwise_gelu_tanh>(
          result, self, 0.0f, 0.0f);
    } else {
      std::cout << "--------onednn go to erf path------" << std::endl;
      xpu::oneDNN::eltwise<dnnl::algorithm::eltwise_gelu_erf>(
          result, self, 0.0f, 0.0f);
    }
    return result;
    // if (xpu::oneDNN::is_onednn_layout(self) &&
    //     xpu::oneDNN::eltwise_forward_valid(self)) {
    //   xpu::oneDNN::eltwise<dnnl::algorithm::eltwise_gelu_erf>(
    //       result, self, 0.0f, 0.0f);
    //   return result;
  } else {
    auto _self = to_plain_if_needed(self);
    if (!result.defined()) {
      result = at::empty_like(_self);
    }
    auto iter = TensorIterator::unary_op(result, _self);

    if (approximate_gelu == at::native::GeluType::Tanh) {
      std::cout << "--------go to tanh path------" << std::endl;
      IPEX_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::BFloat16,
          at::ScalarType::Half,
          iter.dtype(),
          "gelu",
          [&]() {
            dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t self) -> scalar_t {
              return impl::gelu_tanh_forward<scalar_t>(self);
            });
          });
    } else {
      std::cout << "--------go to erf path------" << std::endl;
      IPEX_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::BFloat16,
          at::ScalarType::Half,
          iter.dtype(),
          "gelu",
          [&]() {
            dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t self) -> scalar_t {
              return impl::gelu_erf_forward<scalar_t>(self);
            });
          });
    }
    return result;
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
  std::cout << "--------gelu backward------" << std::endl;
  if (IPEX_ANY(xpu::oneDNN::is_onednn_layout, grad, self) &&
      IPEX_ALL(xpu::oneDNN::eltwise_backward_valid, grad, self)) {
    xpu::oneDNN::eltwise_backward<dnnl::algorithm::eltwise_gelu_erf>(
        grad_input, self, grad, 0.0f, 0.0f);
    return grad_input;
  } else {
    auto _self = to_plain_if_needed(self);
    auto _grad = to_plain_if_needed(grad);
    if (!grad_input.defined()) {
      grad_input = at::empty_like(_self);
    }
    auto iter = TensorIterator::binary_op(grad_input, _grad, _self);
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        iter.dtype(),
        "gelu_backward",
        [&]() {
          dpcpp_kernel_with_scalars(
              iter, [=](scalar_t grad, scalar_t self) -> scalar_t {
                return impl::gelu_erf_backward<scalar_t>(grad, self);
              });
        });
    return grad_input;
  }
}

Tensor gelu_backward(const Tensor& grad, const Tensor& self) {
  Tensor result;
  return gelu_backward_out(grad, self, "none", result);
}

Tensor& silu_out(const Tensor& self, Tensor& output) {
  return impl::silu_out_kernel(self, output);
}

Tensor& silu_backward_out(
    const Tensor& grad_output,
    const Tensor& output,
    Tensor& grad_input) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      output.scalar_type(),
      "silu_backward_out",
      [&]() {
        impl::silu_backward_kernel<scalar_t>(grad_input, grad_output, output);
      });
  return grad_input;
}

at::Tensor& mish_out(const at::Tensor& self, at::Tensor& out) {
  if (xpu::oneDNN::is_onednn_layout(self) &&
      xpu::oneDNN::eltwise_forward_valid(self)) {
    xpu::oneDNN::eltwise<dnnl::algorithm::eltwise_mish>(out, self, 0.0f, 0.0f);
    return out;
  } else {
    auto _self = to_plain_if_needed(self);
    if (!out.defined()) {
      out = at::empty_like(_self);
    }
    auto iter = TensorIterator::unary_op(out, _self);
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        iter.dtype(),
        "mish",
        [&]() {
          dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t self) -> scalar_t {
            return impl::mish_forward<scalar_t>(self);
          });
        });
    return out;
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
