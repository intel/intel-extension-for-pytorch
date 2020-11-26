#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/native/Activation.h>

#include <core/TensorImplUtils.h>
#include <core/ApplyUtils.h>
#include <core/Generator.h>
#include <core/DPCPP.h>

#include <utils/Numerics.h>
#include <utils/ATDispatch.h>

#include "Eltwise.hpp"
#include "Loops.h"
#include "Random.h"

using namespace at::dpcpp::detail;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
static inline bool is_contiguous(const int64_t* strides) {
  return strides[0] == sizeof(scalar_t) && strides[1] == sizeof(scalar_t) &&
      strides[2] == sizeof(scalar_t);
}

template <typename scalar_t>
static void dpcpp_threshold_kernel(TensorIterator& iter) {
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    dpcpp_eltwise_backward<dnnl::algorithm::eltwise_relu>(
        data[0], data[1], data[2], n, 0.0f, 0.0f);
  };
  iter.serial_for_each(loop, {0L, iter.numel()});
}

// Note: dpcpp compiler does not support uname type in template.
class SyclOpThreshold {};

static void threshold_kernel(
    TensorIterator& iter,
    Scalar threshold_scalar,
    Scalar value_scalar) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "threshold",
      [&] {
        scalar_t threshold = threshold_scalar.to<scalar_t>();
        scalar_t value = value_scalar.to<scalar_t>();
        bool all_contiguous = true;
        for (int i = 0; i < iter.ntensors(); i++) {
          all_contiguous = all_contiguous && iter.tensor(i).is_contiguous();
        }

        if (threshold == 0 && value == 0 && all_contiguous && iter.dtype() != ScalarType::BFloat16
            /*is_contiguous<scalar_t>(iter.get_strides().data())*/) {
          dpcpp_threshold_kernel<scalar_t>(iter);
        } else {
          dpcpp_kernel_for_tensor_iter<SyclOpThreshold>(
              iter, [=](scalar_t x, scalar_t other) -> scalar_t {
                return x <= threshold ? value : other;
              });
        }
      });
}

template <typename scalar_t>
class rrulu_updateOutput_dpcpp_inplace_kernel{};

template <typename scalar_t>
class rrulu_updateOutput_dpcpp_kernel{};

template <typename T>
struct RReLUUpdateOutputEvalOp
{
  RReLUUpdateOutputEvalOp(T negSlope) : negSlope_(negSlope) {}

  void operator()(T& in) const {
    in = (in <= 0) ? in * negSlope_ : in;
  }

  void operator()(T& out, T& in) const {
    out = (in <= 0) ? in * negSlope_ : in;
  }

  const T negSlope_;
};

template <typename T>
struct RReLUupdateGradInputEvalOp
{
  const T negSlope_;

  RReLUupdateGradInputEvalOp(T negSlope) : negSlope_(negSlope) {}

  void operator()(T& gradOut, T& in) const {
    gradOut = (in <= 0) ? gradOut * negSlope_ : gradOut;
  }

  void operator()(T& gradIn, T& gradOut, T& in) const {
    gradIn = (in <= 0) ? gradOut * negSlope_ : gradOut;
  }
};


template <typename scalar_t>
static void RReLU_updateOutput(
  const Tensor& input,
  Tensor& output,
  const Tensor& noise,
  double lower,
  double upper,
  bool train,
  bool inplace,
  Generator* generator)
{
  auto gen = at::get_generator_or_default<at::DPCPPGenerator>(
      generator, getDefaultDPCPPGenerator());
  if(train){
    auto input_ = input.contiguous();
    noise.resize_as_(input_);

    std::pair<uint64_t, uint64_t> seeds;
    {
      // See Note [Acquire lock when using random generators]
      // this philox_engine_inputs('1') is aligned with Distribution.cpp,
      // yet they use '((n - 1) / (BLOCK_SIZE * grid.x) + 1) * curand4_engine_calls'
      // in the same place.
      std::lock_guard<std::mutex> lock(gen->mutex_);
      seeds = gen->philox_engine_inputs(1);
    }
    if (inplace)
    {
      auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
      auto total_threads = input_.numel();

      auto cgf = DPCPP_Q_CGF(cgh) {
        auto in_data = get_buffer<dpcpp_r_mode>(cgh, input_.data_ptr<scalar_t>());
        auto noise_data = get_buffer<dpcpp_discard_w_mode>(cgh, noise.data_ptr<scalar_t>());
        cgh.parallel_for<rrulu_updateOutput_dpcpp_inplace_kernel<scalar_t>>(
          DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId){
            auto in_ptr = get_pointer(in_data);
            auto noise_ptr = get_pointer(noise_data);
            auto id = itemId.get_id(0);
            auto linear_id = itemId.get_linear_id();

            RandomState<Philox4_32_10> state(seeds.first, linear_id, seeds.second);

            if(in_ptr[id]<=0)
              {
                double rand = state.uniform<double>();
                scalar_t r = ScalarConvert<double, scalar_t>::to(rand * (upper-lower) + lower);
                in_ptr[id] = static_cast<scalar_t>(in_ptr[id]) * r;
                noise_ptr[id] = r;
              }
              else
              {
                noise_ptr[id]=1;
              }
          });
      };
      DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
      output.set_(input_);
    }
    else
    {
      output.resize_as_(input_);

      auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
      auto total_threads = input_.numel();

      auto cgf = DPCPP_Q_CGF(cgh) {
        auto in_data = get_buffer<dpcpp_r_mode>(cgh, input_.data_ptr<scalar_t>());
        auto out_data = get_buffer<dpcpp_discard_w_mode>(cgh, output.data_ptr<scalar_t>());
        auto noise_data = get_buffer<dpcpp_discard_w_mode>(cgh, noise.data_ptr<scalar_t>());
        cgh.parallel_for<rrulu_updateOutput_dpcpp_kernel<scalar_t>>(
          DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId){
            auto in_ptr = get_pointer(in_data);
            auto out_ptr = get_pointer(out_data);
            auto noise_ptr = get_pointer(noise_data);
            auto id = itemId.get_id(0);
            auto linear_id = itemId.get_linear_id();

            RandomState<Philox4_32_10> state(seeds.first, linear_id, seeds.second);

            if(in_ptr[id]<=0)
              {
                double rand = state.uniform<double>();
                scalar_t r = ScalarConvert<double, scalar_t>::to(rand * (upper-lower) + lower);
                out_ptr[id] = static_cast<scalar_t>(in_ptr[id]) * r;
                noise_ptr[id] = r;
              }
              else
              {
                out_ptr[id] = in_ptr[id];
                noise_ptr[id] = 1;
              }
          });
      };
      DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
    }
  }
  else{
    const scalar_t negSlope = ScalarConvert<double, scalar_t>::to((lower + upper) / 2);
    if (inplace)
    {
      at::dpcpp::DPCPP_tensor_apply1<scalar_t>(
        input, RReLUUpdateOutputEvalOp<scalar_t>(negSlope));
      output.set_(input);
    }
    else
    {
      output.resize_as_(input);
      at::dpcpp::DPCPP_tensor_apply2<scalar_t, scalar_t>(
        output, input, RReLUUpdateOutputEvalOp<scalar_t>(negSlope));
    }
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
  bool inplace)
{
  TORCH_CHECK(
    input.numel() == gradOutput.numel(),
    "input and gradOutput have different number of elements");
  if(train && upper - lower > 1E-6)
  {
    if(inplace)
    {
      gradOutput.mul_(noise);
      gradInput.set_(gradOutput);
    }
    else
    {
      gradInput.resize_as_(input);
      mul_out(gradInput, gradOutput, noise);
    }
  }
  else
  {
    const scalar_t negSlope = ScalarConvert<double, scalar_t>::to((lower + upper) / 2);
    if (inplace)
    {
      at::dpcpp::DPCPP_tensor_apply2<scalar_t, scalar_t>(
        gradOutput, input, RReLUupdateGradInputEvalOp<scalar_t>(negSlope));
      gradInput.set_(gradOutput);
    }
    else
    {
      gradInput.resize_as_(input);
      at::dpcpp::DPCPP_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        gradInput, gradOutput, input, RReLUupdateGradInputEvalOp<scalar_t>(negSlope));
    }
  }
}

/* prelu forward */
template <typename scalar_t>
class prelu_dpcpp_kernel_share_weights{};

template <typename scalar_t>
void inline prelu_kernel_share_weights(
  Tensor& result,
  const Tensor& input,
  const Tensor& weight) {
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  auto total_threads = input.numel();
  auto weight_val = weight.data_ptr<scalar_t>()[0];
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto out_data = get_buffer<dpcpp_discard_w_mode>(cgh, result.data_ptr<scalar_t>());
    auto in_data = get_buffer<dpcpp_r_mode>(cgh, input.data_ptr<scalar_t>());

    cgh.parallel_for<prelu_dpcpp_kernel_share_weights<scalar_t>>(
      DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId){
        auto out_ptr = get_pointer(out_data);
        auto in_ptr = get_pointer(in_data);
        auto id = itemId.get_id(0);
        out_ptr[id] = (in_ptr[id] >= 0) ? in_ptr[id] : weight_val * static_cast<scalar_t>(in_ptr[id]);
      });
  };
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t>
class prelu_dpcpp_kernel_multi_weights{};

template <typename scalar_t>
void inline prelu_kernel_multi_weights(
  Tensor& result,
  const Tensor& input,
  const Tensor& weight,
  int64_t input_dim0_size,
  int64_t channel_size,
  int64_t input_stride0,
  int64_t input_stride1) {

  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  auto total_threads = input.numel();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto out_data = get_buffer<dpcpp_discard_w_mode>(cgh, result.data_ptr<scalar_t>());
    auto in_data = get_buffer<dpcpp_r_mode>(cgh, input.data_ptr<scalar_t>());
    auto weight_data = get_buffer<dpcpp_r_mode>(cgh, weight.data_ptr<scalar_t>());
    cgh.parallel_for<prelu_dpcpp_kernel_multi_weights<scalar_t>>(
      DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId){
        auto out_ptr = get_pointer(out_data);
        auto in_ptr = get_pointer(in_data);
        auto weight_ptr = get_pointer(weight_data);
        auto id = itemId.get_id(0);

        int64_t channel = (id % input_stride0) / input_stride1;
        scalar_t input_data_val = in_ptr[id];
        out_ptr[id] = (input_data_val > 0) ? input_data_val : static_cast<scalar_t>(weight_ptr[channel]) * input_data_val;
      });
  };
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}


/* prelu backward */
template <typename scalar_t>
class prelu_backward_dpcpp_kernel_share_weights{};

template <typename scalar_t>
void inline prelu_backward_kernel_share_weights(
  const Tensor& input,
  const Tensor& weight,
  const Tensor& grad_out,
  Tensor& input_grad,
  Tensor& weight_grad_collector) {

  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  auto total_threads = input_grad.numel();
  auto weight_val = weight.data_ptr<scalar_t>()[0];

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_grad_data = get_buffer<dpcpp_discard_w_mode>(cgh, input_grad.data_ptr<scalar_t>());
    auto weight_grad_collector_data = get_buffer<dpcpp_discard_w_mode>(cgh, weight_grad_collector.data_ptr<scalar_t>());
    auto in_data = get_buffer<dpcpp_r_mode>(cgh, input.data_ptr<scalar_t>());
    auto grad_out_data = get_buffer<dpcpp_r_mode>(cgh, grad_out.data_ptr<scalar_t>());

    cgh.parallel_for<prelu_backward_dpcpp_kernel_share_weights<scalar_t>>(
      DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId){
        auto in_grad_ptr = get_pointer(in_grad_data);
        auto weight_grad_collector_ptr = get_pointer(weight_grad_collector_data);
        auto in_ptr = get_pointer(in_data);
        auto grad_out_ptr = get_pointer(grad_out_data);
        auto id = itemId.get_id(0);

        in_grad_ptr[id] = (in_ptr[id] > 0) ? grad_out_ptr[id] : weight_val * static_cast<scalar_t>(grad_out_ptr[id]);
        weight_grad_collector_ptr[id] = (in_ptr[id] > 0) ? scalar_t(0) : static_cast<scalar_t>(in_ptr[id]) * static_cast<scalar_t>(grad_out_ptr[id]);
      });
  };
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t>
class prelu_backward_dpcpp_kernel_multi_weights{};

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

  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  auto total_threads = input.numel();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = get_buffer<dpcpp_r_mode>(cgh, input.data_ptr<scalar_t>());
    auto weight_data = get_buffer<dpcpp_r_mode>(cgh, weight.data_ptr<scalar_t>());
    auto gred_out_data = get_buffer<dpcpp_r_mode>(cgh, grad_out.data_ptr<scalar_t>());
    auto in_grad_data = get_buffer<dpcpp_discard_w_mode>(cgh, input_grad.data_ptr<scalar_t>());
    auto weight_grad_collector_data = get_buffer<dpcpp_discard_w_mode>(cgh, weight_grad_collector.data_ptr<scalar_t>());
    cgh.parallel_for<prelu_backward_dpcpp_kernel_multi_weights<scalar_t>>(
      DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId){
        auto in_ptr = get_pointer(in_data);
        auto weight_ptr = get_pointer(weight_data);
        auto grad_out_ptr = get_pointer(gred_out_data);
        auto in_grad_ptr = get_pointer(in_grad_data);
        auto weight_grad_collector_ptr = get_pointer(weight_grad_collector_data);
        auto id = itemId.get_id(0);

        int64_t channel = (id % input_stride0) / input_stride1;
        scalar_t input_data_val = in_ptr[id];
        scalar_t grad_out_data_val = grad_out_ptr[id];        
        in_grad_ptr[id] = (input_data_val > 0) ? grad_out_data_val : static_cast<scalar_t>(weight_ptr[channel]) * grad_out_data_val;
        weight_grad_collector_ptr[id] = (input_data_val > 0) ? scalar_t(0) : input_data_val * grad_out_data_val;
      }
    );
  };
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t>
class gelu_dpcpp_kernel{};

template <typename scalar_t>
void GeluKernelImpl(const Tensor& X, Tensor& Y){
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  auto total_threads = X.numel();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto X_data = get_buffer<dpcpp_r_mode>(cgh, X.data_ptr<scalar_t>());
    auto Y_data = get_buffer<dpcpp_discard_w_mode>(cgh, Y.data_ptr<scalar_t>());

    cgh.parallel_for<gelu_dpcpp_kernel<scalar_t>>(
      DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId){
        auto X_ptr = get_pointer(X_data);
        auto Y_ptr = get_pointer(Y_data);
        auto id = itemId.get_id(0);

        Y_ptr[id] = DPCPP::erf(X_ptr[id] * M_SQRT1_2);
        Y_ptr[id] = (static_cast<scalar_t>(Y_ptr[id]) + scalar_t(1)) * static_cast<scalar_t>(X_ptr[id]) * scalar_t(0.5);
      }
    );
  };
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

template <typename scalar_t>
class gelu_backward_dpcpp_kernel{};

template <typename scalar_t>
void GeluBackwardKernelImpl(
  const Tensor& dY,
  const Tensor& X,
  Tensor& dX){
  auto kAlpha = M_2_SQRTPI * M_SQRT1_2 * scalar_t(0.5);
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  auto total_threads = X.numel();
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto dY_data = get_buffer<dpcpp_r_mode>(cgh, dY.data_ptr<scalar_t>());
    auto X_data = get_buffer<dpcpp_r_mode>(cgh, X.data_ptr<scalar_t>());
    auto dX_data = get_buffer<dpcpp_discard_w_mode>(cgh, dX.data_ptr<scalar_t>());
    
    cgh.parallel_for<gelu_backward_dpcpp_kernel<scalar_t>>(
      DPCPP::range<1>(total_threads), [=](DPCPP::item<1> itemId){
        auto dY_ptr = get_pointer(dY_data);
        auto X_ptr = get_pointer(X_data);
        auto dX_ptr = get_pointer(dX_data);
        auto id = itemId.get_id(0);

        dX_ptr[id] = Numerics<scalar_t>::exp(-scalar_t(0.5) * static_cast<scalar_t>(X_ptr[id]) * static_cast<scalar_t>(X_ptr[id]));
        dX_ptr[id] = dY_ptr[id] * 
        (scalar_t(0.5) * (scalar_t(1) + DPCPP::erf(X_ptr[id] * M_SQRT1_2)) + X_ptr[id] * kAlpha * dX_ptr[id]);
      }
    );
  };
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

} // namespace impl

Tensor relu(const Tensor& self) {
  Tensor result;
  at::dpcpp::dpcpp_eltwise<dnnl::algorithm::eltwise_relu>(result, self, 0.0f, 0.0f);
  return result;
}

Tensor& relu_(Tensor& self) {
  at::dpcpp::dpcpp_eltwise<dnnl::algorithm::eltwise_relu>(self, self, 0.0f, 0.0f);
  return self;
}

static Tensor threshold_out(
    optional<Tensor> opt_result,
    const Tensor& self,
    Scalar threshold,
    Scalar value,
    const Tensor& other) {
  Tensor result = opt_result.value_or(Tensor());
  auto iter = TensorIterator::binary_op(result, self, other);
  impl::threshold_kernel(iter, threshold, value);
  return iter.output();
}

Tensor& threshold_(Tensor& self, Scalar threshold, Scalar value) {
  threshold_out(make_optional(self), self, threshold, value, self);
  return self;
}

Tensor threshold(const Tensor& self, Scalar threshold, Scalar value) {
  return threshold_out(nullopt, self, threshold, value, self);
}

Tensor threshold_out(
    Tensor& result,
    const Tensor& self,
    Scalar threshold,
    Scalar value) {
  threshold_out(make_optional(result), self, threshold, value, self);
  return result;
}

Tensor threshold_backward(
    const Tensor& grad,
    const Tensor& self,
    Scalar threshold) {
  return threshold_out(nullopt, self, threshold, 0, grad);
}

Tensor rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator){
  auto self_ = self.contiguous();
  Tensor output = at::empty_like(self_);
  auto lower_ = lower.toDouble();
  auto upper_ = upper.toDouble();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "RReLU_updateOutput", [&]() {
    impl::RReLU_updateOutput<scalar_t>(
      self,
      output,
      noise,
      lower_,
      upper_,
      training,
      false,
      generator);
  });
  return output;
}
//TODO: fix const self
Tensor & rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator){
  auto lower_ = lower.toDouble();
  auto upper_ = upper.toDouble();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "RReLU_updateOutput", [&]() {
    impl::RReLU_updateOutput<scalar_t>(
      self,
      self,
      noise,
      lower_,
      upper_,
      training,
      true,
      generator);
  });
  return self;
}

Tensor & rrelu_with_noise_out(Tensor & out, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator){
  auto lower_ = lower.toDouble();
  auto upper_ = upper.toDouble();
  IPEX_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "RReLU_updateOutput", [&]() {
    impl::RReLU_updateOutput<scalar_t>(
      self,
      out,
      noise,
      lower_,
      upper_,
      training,
      false,
      generator);
  });
  return out;
}

Tensor rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, bool self_is_result){
  Tensor grad_input = at::empty_like(grad_output);
  auto lower_ = lower.toDouble();
  auto upper_ = upper.toDouble();
  IPEX_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, self.scalar_type(), "RReLU_updateGradInput", [&]() {
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

Tensor rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator* generator) {
  return at::rrelu_with_noise(self, at::empty_like(self), lower, upper, training, generator);
}

Tensor & rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator* generator) {
  return at::rrelu_with_noise_(self, at::empty_like(self), lower, upper, training, generator);
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
    IPEX_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "prelu", [&] {
      impl::prelu_kernel_share_weights<scalar_t>(result, input, weight);
    });
  }
  else {
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
    TORCH_CHECK(channel_size == weight_num,
      "Mismatch of parameter numbers and input channel size. Found parameter numbers = ", weight_num,
      " and channel size = ", channel_size, ".");

    IPEX_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "prelu", [&] {
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

std::tuple<Tensor, Tensor> prelu_backward(const Tensor& grad_out_, const Tensor& self, const Tensor& weight_) {
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
    IPEX_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "prelu_backward", [&] {
      impl::prelu_backward_kernel_share_weights<scalar_t>(input, weight, grad_out, input_grad, weight_grad_collector);
    });
    //fix me: fill_() returns RuntimeError when input weight_grad_collector.sum() is without '.item()'
    weight_grad.fill_(weight_grad_collector.sum().item());
  }
  else {
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
    TORCH_CHECK(channel_size == weight_num,
      "Mismatch of parameter numbers and input channel size. Found parameter numbers = ", weight_num,
      " and channel size = ", channel_size, ".");

    IPEX_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "prelu_backward", [&] {
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
      for(int64_t i = 2; i < dims; i++) reduce_dims.push_back(i);
    }
    weight_grad = weight_grad_collector.sum(reduce_dims);
  }
  return std::tuple<Tensor, Tensor>{input_grad, weight_grad};
}

DPCPP_DEF_K1(DPCPPOpHardShrink);
Tensor hardshrink(const Tensor& self, Scalar lambd_) {
  auto out_tensor = at::empty_like(self);

  auto iter = TensorIterator();
  iter.add_output(out_tensor);
  iter.add_input(self);
  iter.build();

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardshrink",
      [&] {
        auto lambd = lambd_.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter<DPCPP_K(DPCPPOpHardShrink)>(
            iter, [=](scalar_t x) -> scalar_t {
              return (x >= -lambd && x <= lambd) ? scalar_t(0) : x;
            });
      });
  return out_tensor;
}

DPCPP_DEF_K1(DPCPPOpHardShrinkBackward);
Tensor hardshrink_backward(
    const Tensor& grad,
    const Tensor& self,
    Scalar lambd_) {
  auto out_tensor = at::empty_like(grad);

  auto iter = TensorIterator();
  iter.add_output(out_tensor);
  iter.add_input(grad);
  iter.add_input(self);
  iter.build();

  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, self.scalar_type(), "hardshrink_backward", [&] {
        auto lambd = lambd_.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter<DPCPP_K(DPCPPOpHardShrinkBackward)>(
            iter, [=](scalar_t grad_output, scalar_t x) -> scalar_t {
              return (x >= -lambd && x <= lambd) ? scalar_t(0) : grad_output;
            });
      });
  return out_tensor;
}

Tensor gelu(const Tensor & self){
  auto self_ = self.contiguous();
  Tensor Y = at::empty_like(self_);
  IPEX_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self_.scalar_type(), "GeluKernelImpl", [&](){
    impl::GeluKernelImpl<scalar_t>(self_, Y);
  });
  return Y;
}

Tensor gelu_backward(const Tensor & grad, const Tensor & self){
  auto self_ = self.contiguous();
  Tensor dX = at::empty_like(self_);
  IPEX_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, self_.scalar_type(), "GeluBackwardKernelImpl", [&](){
    impl::GeluBackwardKernelImpl<scalar_t>(grad, self_, dX);
  });
  return dX;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
