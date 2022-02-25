#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>

#include <core/Generator.h>
#include <core/TensorImplUtils.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include <utils/Macros.h>
#include "comm/ApplyUtils.h"

#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"

#include <oneDNN/oneDNN.h>
#include "Loops.h"
#include "Random.h"

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
            DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId) {
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
            DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId) {
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
      at::AtenIpexTypeXPU::mul_out(gradInput, gradOutput, noise);
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
void inline prelu_kernel_share_weights(
    Tensor& result,
    const Tensor& input,
    const Tensor& weight) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = input.numel();
  auto weight_val = weight.data_ptr<scalar_t>();
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto out_data = result.data_ptr<scalar_t>();
    auto in_data = input.data_ptr<scalar_t>();

    cgh.parallel_for(
        DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId) {
          auto out_ptr = out_data;
          auto in_ptr = in_data;
          auto id = itemId.get_id(0);
          out_ptr[id] = (in_ptr[id] >= 0)
              ? in_ptr[id]
              : (*weight_val) * static_cast<scalar_t>(in_ptr[id]);
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t>
void inline prelu_kernel_multi_weights(
    Tensor& result,
    const Tensor& input,
    const Tensor& weight,
    int64_t input_dim0_size,
    int64_t channel_size,
    int64_t input_stride0,
    int64_t input_stride1) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto total_threads = input.numel();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto out_data = result.data_ptr<scalar_t>();
    auto in_data = input.data_ptr<scalar_t>();
    auto weight_data = weight.data_ptr<scalar_t>();
    cgh.parallel_for(
        DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId) {
          auto out_ptr = out_data;
          auto in_ptr = in_data;
          auto weight_ptr = weight_data;
          auto id = itemId.get_id(0);

          int64_t channel = (id % input_stride0) / input_stride1;
          scalar_t input_data_val = in_ptr[id];
          out_ptr[id] = (input_data_val > 0)
              ? input_data_val
              : static_cast<scalar_t>(weight_ptr[channel]) * input_data_val;
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
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

    cgh.parallel_for(
        DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId) {
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
    cgh.parallel_for(
        DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId) {
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
    Scalar threshold,
    Scalar value,
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
inline scalar_t relu_forward(scalar_t self) {
  return self > 0 ? self : static_cast<scalar_t>(0);
}

template <typename scalar_t>
inline scalar_t gelu_erf_forward(scalar_t self) {
  using accscalar_t = acc_type<scalar_t>;
  auto v = static_cast<accscalar_t>(self) * M_SQRT1_2;
  return (scalar_t)(M_SQRT1_2 * v * (1.0 + Numerics<accscalar_t>::erf(v)));
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
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
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

Tensor& threshold_(Tensor& self, Scalar threshold, Scalar value) {
  impl::threshold_out(make_optional(self), self, threshold, value, self);
  return self;
}

Tensor threshold(const Tensor& self, Scalar threshold, Scalar value) {
  return impl::threshold_out(nullopt, self, threshold, value, self);
}

Tensor& threshold_out(
    Tensor& result,
    const Tensor& self,
    Scalar threshold,
    Scalar value) {
  impl::threshold_out(make_optional(result), self, threshold, value, self);
  return result;
}

Tensor threshold_backward(
    const Tensor& grad,
    const Tensor& self,
    Scalar threshold) {
  return impl::threshold_out(nullopt, self, threshold, 0, grad);
}

Tensor rrelu_with_noise(
    const Tensor& self,
    const Tensor& noise,
    Scalar lower,
    Scalar upper,
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
    Scalar lower,
    Scalar upper,
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
    Tensor& out,
    const Tensor& self,
    const Tensor& noise,
    Scalar lower,
    Scalar upper,
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
            self, out, noise, lower_, upper_, training, false, generator);
      });
  return out;
}

Tensor rrelu_with_noise_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& noise,
    Scalar lower,
    Scalar upper,
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

Tensor rrelu(
    const Tensor& self,
    Scalar lower,
    Scalar upper,
    bool training,
    c10::optional<Generator> generator) {
  return at::rrelu_with_noise(
      self, at::empty_like(self), lower, upper, training, generator);
}

Tensor& rrelu_(
    Tensor& self,
    Scalar lower,
    Scalar upper,
    bool training,
    c10::optional<Generator> generator) {
  return at::rrelu_with_noise_(
      self, at::empty_like(self), lower, upper, training, generator);
}

Tensor prelu(const Tensor& self, const Tensor& weight_) {
  auto input = self.contiguous();
  auto weight = weight_.contiguous();

  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());

  int64_t weight_num = weight.numel();
  Tensor result = at::empty_like(input);
  auto strides = input.strides();

  if (weight_num == 1) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "prelu",
        [&] {
          impl::prelu_kernel_share_weights<scalar_t>(result, input, weight);
        });
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
        "prelu",
        [&] {
          impl::prelu_kernel_multi_weights<scalar_t>(
              result,
              input,
              weight,
              input_dim0_size,
              channel_size,
              input_stride0,
              input_stride1);
        });
  }
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

Tensor hardshrink(const Tensor& self, Scalar lambd_) {
  auto out_tensor = at::empty_like(self);

  auto iter =
      TensorIteratorConfig().add_output(out_tensor).add_input(self).build();

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardshrink",
      [&] {
        auto lambd = lambd_.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t x) -> scalar_t {
          return (x >= -lambd && x <= lambd) ? scalar_t(0) : x;
        });
      });
  return out_tensor;
}

Tensor hardshrink_backward(
    const Tensor& grad,
    const Tensor& self,
    Scalar lambd_) {
  auto out_tensor = at::empty_like(grad);

  TensorIterator iter = TensorIteratorConfig()
                            .add_output(out_tensor)
                            .add_input(grad)
                            .add_input(self)
                            .build();

  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, self.scalar_type(), "hardshrink_backward", [&] {
        auto lambd = lambd_.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(
            iter, [=](scalar_t grad_output, scalar_t x) -> scalar_t {
              return (x >= -lambd && x <= lambd) ? scalar_t(0) : grad_output;
            });
      });
  return out_tensor;
}

Tensor gelu(const Tensor& self) {
  if (xpu::oneDNN::is_onednn_layout(self) &&
      xpu::oneDNN::eltwise_forward_valid(self)) {
    Tensor result;
    xpu::oneDNN::eltwise<dnnl::algorithm::eltwise_gelu_erf>(
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
        "gelu",
        [&]() {
          dpcpp_kernel_for_tensor_iter(iter, [=](scalar_t self) -> scalar_t {
            return impl::gelu_erf_forward<scalar_t>(self);
          });
        });
    return result;
  }
}

Tensor gelu_backward(const Tensor& grad, const Tensor& self) {
  if (IPEX_ANY(xpu::oneDNN::is_onednn_layout, grad, self) &&
      IPEX_ALL(xpu::oneDNN::eltwise_backward_valid, grad, self)) {
    Tensor dX;
    xpu::oneDNN::eltwise_backward<dnnl::algorithm::eltwise_gelu_erf>(
        dX, self, grad, 0.0f, 0.0f);
    return dX;
  } else {
    auto _self = to_plain_if_needed(self);
    auto _grad = to_plain_if_needed(grad);
    auto dX = at::empty_like(_self);
    auto iter = TensorIterator::binary_op(dX, _grad, _self);
    IPEX_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16, iter.dtype(), "gelu_backward", [&]() {
          dpcpp_kernel_with_scalars(
              iter, [=](scalar_t grad, scalar_t self) -> scalar_t {
                return impl::gelu_erf_backward<scalar_t>(grad, self);
              });
        });
    return dX;
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
