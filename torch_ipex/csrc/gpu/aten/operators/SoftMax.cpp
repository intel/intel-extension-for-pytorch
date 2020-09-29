#include <ATen/ATen.h>
#include <utils/AccumulateType.h>
#include <utils/ATDispatch.h>
#include <core/Context.h>
#include <core/Memory.h>
#include <core/detail/TensorInfo.h>

#include <utils/Numerics.h>
#include <operators/Reduce.h>

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename...>
class SpatialSoftmaxForwardKernelName {};

template <typename...>
class SpatialSoftmaxBackwardKernelName {};

template <typename T, typename AccumT, typename OutT>
struct LogSoftMaxForwardEpilogue {
  LogSoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
      : logsum(max_input + Numerics<AccumT>::log(sum)) {}

  OutT operator()(T input) const {
    return static_cast<OutT>(input - logsum);
  }
  const AccumT logsum;
};

template <typename T, typename AccumT, typename OutT>
struct LogSoftMaxBackwardEpilogue {
  LogSoftMaxBackwardEpilogue(AccumT sum) : sum(sum) {}

  T operator()(OutT gradOutput, OutT output) const {
    return static_cast<T>(
        static_cast<AccumT>(gradOutput) -
        Numerics<AccumT>::exp(static_cast<AccumT>(output)) * sum);
  }

  const AccumT sum;
};

template <typename T, typename AccumT, typename OutT>
struct SoftMaxForwardEpilogue {
  SoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
      : max_input(max_input), sum(sum) {}

  OutT operator()(T input) const {
    return Numerics<OutT>::exp(static_cast<OutT>(input - max_input)) / sum;
  }

  const AccumT max_input;
  const AccumT sum;
};

template <typename T, typename AccumT, typename OutT>
struct SoftMaxBackwardEpilogue {
  SoftMaxBackwardEpilogue(AccumT sum) : sum(sum) {}

  // XXX: gradOutput that we get here is really gradOutput * output
  // Look for cmul in SoftMax_updateGradInput
  T operator()(OutT gradOutput, OutT output) const {
    return static_cast<T>(gradOutput - output * sum);
  }

  const AccumT sum;
};

// It is naive implementation for the softmax. Not optimized if the dim_size is
// small.
template <
    typename scalar_t,
    typename accscalar_t,
    typename outscalar_t,
    template <typename, typename, typename> class Epilogue>
void SpatialSoftMaxForward(
    outscalar_t* output,
    scalar_t* input,
    dpcpp::detail::TensorInfo<scalar_t, uint64_t> outer_info,
    size_t outer_size,
    size_t dim_size,
    size_t dim_stride) {
  using local_accessor_t = DPCPP::accessor<
      accscalar_t,
      1,
      DPCPP::access::mode::read_write,
      DPCPP::access::target::local>;
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  size_t local_size = dpcppMaxWorkGroupSize(dpcpp_queue);
  local_size = std::min(local_size, dim_size);
  size_t global_size = outer_size * local_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = get_buffer<dpcpp_r_mode>(cgh, input);
    auto out_data = get_buffer<dpcpp_discard_w_mode>(cgh, output);
    auto local_acc_max = local_accessor_t(local_size, cgh);
    auto local_acc_sum = local_accessor_t(local_size, cgh);
    cgh.parallel_for<SpatialSoftmaxForwardKernelName<
        scalar_t,
        Epilogue<scalar_t, accscalar_t, outscalar_t>>>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(global_size), DPCPP::range<1>(local_size)),
        [=](DPCPP::nd_item<1> item_id) {
          size_t local_id = item_id.get_local_id(0);
          auto group_id = item_id.get_group(0);
          auto data_offset =
              dpcpp::detail::IndexToOffset<scalar_t, uint64_t>::get(
                  group_id, outer_info);
          auto in_ptr = get_pointer(in_data) + data_offset;
          auto out_ptr = get_pointer(out_data)+ data_offset;
          // get max
          auto max_input = in_ptr[0];
          for (uint32_t i = local_id; i < dim_size; i += local_size) {
            max_input =
                Numerics<scalar_t>::max(max_input, in_ptr[i * dim_stride]);
          }
          // to accscalar_t
          local_acc_max[local_id] = static_cast<accscalar_t>(max_input);

          at::dpcpp::reduce(item_id, local_acc_max, [](accscalar_t a, accscalar_t b) {
            return Numerics<accscalar_t>::max(a, b);
          });

          // get sum
          auto sum_input = static_cast<accscalar_t>(0);
          for (size_t i = local_id; i < dim_size; i += local_size) {
            // (NOTE) This arithmetic convension is to avoid dp_global_ptr cast
            auto in_data = static_cast<accscalar_t>(in_ptr[i * dim_stride]);
            sum_input += Numerics<accscalar_t>::exp(in_data - local_acc_max[0]);
          }
          local_acc_sum[local_id] = sum_input;

          at::dpcpp::reduce(item_id, local_acc_sum, [](accscalar_t a, accscalar_t b) {
            return a + b;
          });

          Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(
              local_acc_max[0], local_acc_sum[0]);

          for (size_t i = local_id; i < dim_size; i += local_size) {
            out_ptr[i * dim_stride] = epilogue(in_ptr[i * dim_stride]);
          }
        });
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

// It is naive implementation for the softmax. Not optimized if the dim_size is
// small.
template <
    typename scalar_t,
    typename accscalar_t,
    typename outscalar_t,
    template <typename, typename, typename> class Epilogue>
void SpatialSoftMaxBackward(
    scalar_t* gradInput,
    outscalar_t* output,
    outscalar_t* gradOutput,
    dpcpp::detail::TensorInfo<scalar_t, uint64_t> outer_info,
    size_t outer_size,
    size_t dim_size,
    size_t dim_stride) {
  using local_accessor_t = DPCPP::accessor<
      accscalar_t,
      1,
      DPCPP::access::mode::read_write,
      DPCPP::access::target::local>;
  auto& dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  size_t local_size = dpcppMaxWorkGroupSize(dpcpp_queue);
  local_size = std::min(local_size, dim_size);
  size_t global_size = outer_size * local_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto gradInput_data = get_buffer<dpcpp_discard_w_mode>(cgh, gradInput);
    auto output_data = get_buffer<dpcpp_r_mode>(cgh, output);
    auto gradOutput_data = get_buffer<dpcpp_r_mode>(cgh, gradOutput);
    auto local_acc_sum = local_accessor_t(local_size, cgh);
    cgh.parallel_for<SpatialSoftmaxBackwardKernelName<
        scalar_t,
        Epilogue<scalar_t, accscalar_t, outscalar_t>>>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(global_size), DPCPP::range<1>(local_size)),
        [=](DPCPP::nd_item<1> item_id) {
          size_t local_id = item_id.get_local_id(0);
          auto group_id = item_id.get_group(0);
          auto data_offset =
              dpcpp::detail::IndexToOffset<outscalar_t, uint64_t>::get(
                  group_id, outer_info);
          auto gradInput_ptr = get_pointer(gradInput_data) + data_offset;
          auto output_ptr = get_pointer(output_data) + data_offset;
          auto gradOutput_ptr = get_pointer(gradOutput_data) + data_offset;

          auto thread_sum = static_cast<accscalar_t>(0);
          for (size_t i = local_id; i < dim_size; i += local_size) {
            thread_sum +=
                static_cast<accscalar_t>(gradOutput_ptr[i * dim_stride]);
          }
          local_acc_sum[local_id] = thread_sum;

          at::dpcpp::reduce(item_id, local_acc_sum, [](accscalar_t a, accscalar_t b) {
            return a + b;
          });

          auto sum_k = local_acc_sum[0];
          Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(sum_k);
          for (size_t i = local_id; i < dim_size; i += local_size) {
            gradInput_ptr[i * dim_stride] = epilogue(
                gradOutput_ptr[i * dim_stride], output_ptr[i * dim_stride]);
          }
        });
  };

  // launch kernel
  DPCPP_Q_ASYNC_SUBMIT(dpcpp_queue, cgf);
}

} // namespace impl

template <template <typename, typename, typename> class Epilogue>
Tensor host_softmax(
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float) {
  AT_ASSERTM(
      !half_to_float,
      "softmax with half to float conversion is not supported on DPCPP");
  auto input = input_.contiguous();
  Tensor output = at::native::empty_like(input);
  if (input.dim() == 0)
    input = input.view(1);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  TORCH_CHECK(
      dim >= 0 && dim < input.dim(),
      "** dpcpp dim must be non-negative and less than input dimensions");

  if (input.numel() > 0) {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        input.scalar_type(),
        "host_softmax",
        [&] {
          auto dim_stride = input.stride(dim);
          auto dim_size = input.size(dim);
          auto outer_numel = input.numel() / dim_size;
          dpcpp::detail::TensorInfo<scalar_t, uint64_t> outer_info =
              dpcpp::detail::getTensorInfo<scalar_t, uint64_t>(input);
          outer_info.reduceDim(dim);
          outer_info.collapseDims();
          using accscalar_t = acc_type<scalar_t>;
          using outscalar_t = scalar_t;
          impl::SpatialSoftMaxForward<
              scalar_t,
              accscalar_t,
              outscalar_t,
              Epilogue>(
              output.data_ptr<outscalar_t>(),
              input.data_ptr<scalar_t>(),
              outer_info,
              outer_numel,
              dim_size,
              dim_stride);
        });
  }
  return output;
}

template <template <typename, typename, typename> class Epilogue>
Tensor host_softmax_backward(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    bool half_to_float) {
  AT_ASSERTM(
      !half_to_float,
      "softmax with half to float conversion is not supported on DPCPP");
  int64_t dim = maybe_wrap_dim(dim_, grad_.dim());
  auto grad = grad_.contiguous();
  Tensor gI = at::empty_like(grad);
  if (grad.dim() == 0)
    grad = grad.view(1);
  TORCH_CHECK(
      dim >= 0 && dim < grad.dim(),
      "dim must be non-negative and less than input dimensions");
  auto output = output_.contiguous();
  if (output.dim() == 0)
    output = output.view(1);
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "host_softmax_backward",
      [&] {
        using accscalar_t = acc_type<scalar_t>;
        using outscalar_t = scalar_t;
        auto dim_stride = output.stride(dim);
        auto dim_size = output.size(dim);
        auto outer_numel = output.numel() / dim_size;
        dpcpp::detail::TensorInfo<outscalar_t, uint64_t> outer_info =
            dpcpp::detail::getTensorInfo<outscalar_t, uint64_t>(output);
        outer_info.reduceDim(dim);
        outer_info.collapseDims();
        impl::SpatialSoftMaxBackward<
            scalar_t,
            accscalar_t,
            outscalar_t,
            Epilogue>(
            gI.data_ptr<scalar_t>(),
            output.data_ptr<outscalar_t>(),
            grad.data_ptr<outscalar_t>(),
            outer_info,
            outer_numel,
            dim_size,
            dim_stride);
      });
  return gI;
}

Tensor _softmax(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float) {
  return host_softmax<impl::SoftMaxForwardEpilogue>(input, dim, half_to_float);
}

Tensor _softmax_backward_data(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  bool half_to_float = grad.scalar_type() != input.scalar_type();
  if (half_to_float) {
    TORCH_CHECK(
        !half_to_float,
        "softmax backward with half to float "
        "conversion is not supported on DPCPP");
  }
  Tensor tmp = grad * output;
  return host_softmax_backward<impl::SoftMaxBackwardEpilogue>(
      tmp, output, dim, half_to_float);
}

Tensor _log_softmax(const Tensor& self, int64_t dim, bool half_to_float) {
  return host_softmax<impl::LogSoftMaxForwardEpilogue>(
      self, dim, half_to_float);
}

Tensor _log_softmax_backward_data(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  bool half_to_float = grad.scalar_type() != input.scalar_type();
  if (half_to_float) {
    TORCH_INTERNAL_ASSERT(
        !half_to_float,
        "softmax with half to float conversion is not supported on DPCPP");
  }
  return host_softmax_backward<impl::LogSoftMaxBackwardEpilogue>(
      grad, output, dim, half_to_float);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
