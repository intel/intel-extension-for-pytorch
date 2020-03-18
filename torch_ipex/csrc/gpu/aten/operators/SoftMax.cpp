#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include <core/Context.h>
#include <core/Memory.h>

using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t, template <typename> class Epilogue>
class SoftmaxForwardKernelName {};

template <typename scalar_t, template <typename> class Epilogue>
class SoftmaxBackwardKernelName {};

template <typename T> struct LogSoftMaxForwardEpilogue {
  LogSoftMaxForwardEpilogue(T max_input, T sum)
      : logsum(max_input + DPCPP::log(static_cast<float>(sum))) {}
  T operator()(T input) const { return static_cast<T>(input - logsum); }
  const T logsum;
};

template <typename T> struct LogSoftMaxBackwardEpilogue {
  LogSoftMaxBackwardEpilogue(T sum) : sum(sum) {}
  T operator()(T gradOutput, T output) const {
    return static_cast<T>(gradOutput -
                          DPCPP::exp(static_cast<T>(output)) * sum);
  }

  const T sum;
};

template <typename T> struct SoftMaxForwardEpilogue {
  SoftMaxForwardEpilogue(T max_input, T sum) : max_input(max_input), sum(sum) {}
  T operator()(T input) const {
    return static_cast<T>(DPCPP::exp(static_cast<float>(input - max_input)) /
                          sum);
  }

  const T max_input;
  const T sum;
};

template <typename T> struct SoftMaxBackwardEpilogue {
  SoftMaxBackwardEpilogue(T sum) : sum(sum) {}
  // XXX: gradOutput that we get here is really gradOutput * output
  // Look for cmul in SoftMax_updateGradInput
  T operator()(T gradOutput, T output) const {
    return static_cast<T>(gradOutput - output * sum);
  }

  const T sum;
};

template <typename scalar_t, template <typename> class Epilogue>
void SoftMaxForward(scalar_t *output, scalar_t *input, int classes,
                    int out_size) {
  static const auto write_mode = DPCPP::access::mode::discard_write;
  static const auto read_mode = DPCPP::access::mode::read;
  using local_accessor_t =
      DPCPP::accessor<scalar_t, 1, DPCPP::access::mode::read_write,
                      DPCPP::access::target::local>;
  auto &dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  int64_t local_size =
      dpcpp_queue.get_device()
          .template get_info<DPCPP::info::device::max_work_group_size>();
  int64_t global_size = out_size * local_size;
  dpcpp_queue.submit([&](DPCPP::handler &cgh) {
    auto in_acc = DPCPPAccessor<read_mode>(cgh, input, out_size * classes *
                                                           sizeof(scalar_t));
    auto out_acc = DPCPPAccessor<write_mode>(cgh, output, out_size * classes *
                                                              sizeof(scalar_t));
    auto local_acc_max = local_accessor_t(local_size, cgh);
    auto local_acc_sum = local_accessor_t(local_size, cgh);
    cgh.parallel_for<SoftmaxForwardKernelName<scalar_t, Epilogue>>(
        DPCPP::nd_range<1>(DPCPP::range<1>(global_size),
                           DPCPP::range<1>(local_size)),
        [=](DPCPP::nd_item<1> item_id) {
          int64_t local_id = item_id.get_local_id(0);
          auto group_id = item_id.get_group(0);
          auto in_ptr =
              in_acc.template get_pointer<scalar_t>() + classes * group_id;
          auto out_ptr =
              out_acc.template get_pointer<scalar_t>() + classes * group_id;
          // get max
          auto max_input = in_ptr[0];
          for (int i = local_id; i < classes; i += local_size) {
            max_input = DPCPP::max(static_cast<float>(max_input),
                                   static_cast<float>(in_ptr[i]));
          }
          local_acc_max[local_id] = max_input;

          for (int i = (local_size >> 1); i > 0; i >>= 1) {
            item_id.barrier(DPCPP::access::fence_space::local_space);
            if (local_id < i)
              local_acc_max[local_id] =
                  DPCPP::max(static_cast<float>(local_acc_max[local_id]),
                             static_cast<float>(local_acc_max[local_id + i]));
          }
          item_id.barrier(DPCPP::access::fence_space::local_space);

          // get sum
          auto sum_input = static_cast<scalar_t>(0);
          for (int i = local_id; i < classes; i += local_size) {
            sum_input += DPCPP::exp(static_cast<float>(in_ptr[i]) -
                                    static_cast<float>(local_acc_max[0]));
          }
          local_acc_sum[local_id] = sum_input;

          for (int i = (local_size >> 1); i > 0; i >>= 1) {
            item_id.barrier(DPCPP::access::fence_space::local_space);
            if (local_id < i)
              local_acc_sum[local_id] += local_acc_sum[local_id + i];
          }
          item_id.barrier(DPCPP::access::fence_space::local_space);
          Epilogue<scalar_t> epilogue(local_acc_max[0], local_acc_sum[0]);

          for (int i = local_id; i < classes; i += local_size) {
            out_ptr[i] = epilogue(in_ptr[i]);
          }
        });
  });
}

template <typename scalar_t, template <typename> class Epilogue>
void SoftMaxBackward(scalar_t *gradInput, scalar_t *output,
                     scalar_t *gradOutput, int classes, int out_size) {
  static const auto write_mode = DPCPP::access::mode::discard_write;
  static const auto read_mode = DPCPP::access::mode::read;
  using local_accessor_t =
      DPCPP::accessor<scalar_t, 1, DPCPP::access::mode::read_write,
                      DPCPP::access::target::local>;
  auto &dpcpp_queue = getCurrentDPCPPStream().dpcpp_queue();
  int64_t local_size =
      dpcpp_queue.get_device()
          .template get_info<DPCPP::info::device::max_work_group_size>();
  int64_t global_size = out_size * local_size;
  dpcpp_queue.submit([&](DPCPP::handler &cgh) {
    auto gradInput_acc = DPCPPAccessor<write_mode>(
        cgh, gradInput, out_size * classes * sizeof(scalar_t));
    auto output_acc = DPCPPAccessor<read_mode>(
        cgh, output, out_size * classes * sizeof(scalar_t));
    auto gradOutput_acc = DPCPPAccessor<read_mode>(
        cgh, gradOutput, out_size * classes * sizeof(scalar_t));
    auto local_acc_sum = local_accessor_t(local_size, cgh);
    cgh.parallel_for<SoftmaxBackwardKernelName<scalar_t, Epilogue>>(
        DPCPP::nd_range<1>(DPCPP::range<1>(global_size),
                           DPCPP::range<1>(local_size)),
        [=](DPCPP::nd_item<1> item_id) {
          int64_t local_id = item_id.get_local_id(0);
          auto group_id = item_id.get_group(0);
          auto gradInput_ptr = gradInput_acc.template get_pointer<scalar_t>() +
                               classes * group_id;
          auto output_ptr =
              output_acc.template get_pointer<scalar_t>() + classes * group_id;
          auto gradOutput_ptr =
              gradOutput_acc.template get_pointer<scalar_t>() +
              classes * group_id;

          auto thread_sum = static_cast<scalar_t>(0);
          for (int64_t i = local_id; i < classes; i += local_size) {
            thread_sum += gradOutput_ptr[i];
          }
          local_acc_sum[local_id] = thread_sum;
          for (int64_t i = (local_size >> 1); i > 0; i >>= 1) {
            item_id.barrier(DPCPP::access::fence_space::local_space);
            if (local_id < i)
              local_acc_sum[local_id] += local_acc_sum[local_id + i];
          }
          auto sum_k = local_acc_sum[0];
          Epilogue<scalar_t> epilogue(sum_k);
          for (int64_t i = local_id; i < classes; i += local_size)
            gradInput_ptr[i] = epilogue(gradOutput_ptr[i], output_ptr[i]);
        });
  });
}

} // namespace impl

template <template <typename> class Epilogue>
Tensor host_softmax(const Tensor &input_, const int64_t dim_,
                    const bool half_to_float) {
  AT_ASSERTM(!half_to_float,
             "softmax with half to float conversion is not supported on DPCPP");
  auto input = input_.contiguous();
  Tensor output = at::empty_like(input);
  if (input.dim() == 0)
    input = input.view(1);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  TORCH_CHECK(
      dim >= 0 && dim < input.dim(),
      "** dpcpp dim must be non-negative and less than input dimensions");

  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);

  if (input.numel() > 0) {
    int64_t inner_size = 1;
    for (int64_t i = 0; i < dim; ++i)
      outer_size *= input.size(i);
    for (int64_t i = dim + 1; i < input.dim(); ++i)
      inner_size *= input.size(i);
    if (inner_size == 1) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          input.scalar_type(), "host_softmax", [&] {
            impl::SoftMaxForward<scalar_t, Epilogue>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                dim_size, outer_size);
          });
    } else {
    }
  }
  return output;
}

template <template <typename> class Epilogue>
Tensor host_softmax_backward(const Tensor &grad_, const Tensor &output_,
                             int64_t dim_, bool half_to_float) {
  AT_ASSERTM(!half_to_float,
             "softmax with half to float conversion is not supported on DPCPP");
  int64_t dim = maybe_wrap_dim(dim_, grad_.dim());
  auto grad = grad_.contiguous();
  Tensor gI = at::empty_like(grad);
  if (grad.dim() == 0)
    grad = grad.view(1);
  TORCH_CHECK(dim >= 0 && dim < grad.dim(),
              "dim must be non-negative and less than input dimensions");
  auto output = output_.contiguous();
  if (output.dim() == 0)
    output = output.view(1);
  int64_t outer_size = 1;
  int64_t dim_size = output.size(dim);
  int64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= output.size(i);
  for (int64_t i = dim + 1; i < output.dim(); ++i)
    inner_size *= output.size(i);
  if (inner_size == 1) {
    AT_DISPATCH_FLOATING_TYPES(gI.scalar_type(), "host_softmax_backward", [&] {
      impl::SoftMaxBackward<scalar_t, Epilogue>(
          gI.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
          grad.data_ptr<scalar_t>(), dim_size, outer_size);
    });
  } else {
  }
  return gI;
}

Tensor _softmax(const Tensor &input, const int64_t dim,
                const bool half_to_float) {
  return host_softmax<impl::SoftMaxForwardEpilogue>(input, dim, half_to_float);
}

Tensor _softmax_backward_data(const Tensor &grad, const Tensor &output,
                              int64_t dim, const Tensor &input) {
  bool half_to_float = grad.scalar_type() != input.scalar_type();
  if (half_to_float) {
    TORCH_CHECK(!half_to_float, "softmax backward with half to float "
                                "conversion is not supported on DPCPP");
  }
  Tensor tmp = grad * output;
  return host_softmax_backward<impl::SoftMaxBackwardEpilogue>(tmp, output, dim,
                                                              half_to_float);
}

Tensor _log_softmax(const Tensor &self, int64_t dim, bool half_to_float) {
  return host_softmax<impl::LogSoftMaxForwardEpilogue>(self, dim,
                                                       half_to_float);
}

Tensor _log_softmax_backward_data(const Tensor &grad, const Tensor &output,
                                  int64_t dim, const Tensor &input) {
  bool half_to_float = grad.scalar_type() != input.scalar_type();
  if (half_to_float) {
    AT_ASSERTM(
        !half_to_float,
        "softmax with half to float conversion is not supported on DPCPP");
  }
  return host_softmax_backward<impl::LogSoftMaxBackwardEpilogue>(
      grad, output, dim, half_to_float);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
