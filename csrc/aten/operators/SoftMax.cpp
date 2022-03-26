#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <core/Memory.h>
#include <core/detail/TensorInfo.h>
#include <intrinsic/intrinsic.h>
#include <oneDNN/oneDNN.h>
#include <runtime/Utils.h>
#include "comm/RegistrationDeclarations.h"

#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Numerics.h"
#include "comm/SimpleReduce.h"

using namespace dnnl;
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

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

// It is naive implementation for the softmax with dim_stride=1.
template <
    typename scalar_t,
    typename accscalar_t,
    typename outscalar_t,
    template <typename, typename, typename>
    class Epilogue>
void softmax_lastdim_fwd_kernel(
    scalar_t* in_data,
    outscalar_t* out_data,
    TensorInfo<scalar_t, uint64_t> outer_info,
    size_t dim_size,
    size_t dim_stride,
    size_t outer_size,
    size_t local_size) {
  RECORD_FUNCTION("softmax_lastdim_fwd_kernel", {});
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  DPCPP::range<1> global_range(outer_size * local_size);
  DPCPP::range<1> local_range(local_size);
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto local_acc_max = dpcpp_local_acc_t<accscalar_t>(local_size, cgh);
    auto local_acc_sum = dpcpp_local_acc_t<accscalar_t>(local_size, cgh);
    cgh.parallel_for(
        DPCPP::nd_range<1>(global_range, local_range),
        [=](DPCPP::nd_item<1> item_id) {
          size_t local_id = item_id.get_local_id(0);
          auto group_id = item_id.get_group(0);
          auto data_offset =
              IndexToOffset<scalar_t, uint64_t>::get(group_id, outer_info);
          auto in_ptr = in_data + data_offset;
          auto out_ptr = out_data + data_offset;

          // get max
          auto max_input = in_ptr[local_id];
          for (uint32_t i = local_id + local_size; i < dim_size;
               i += local_size) {
            max_input = Numerics<scalar_t>::max(max_input, in_ptr[i]);
          }
          // to accscalar_t
          local_acc_max[local_id] = static_cast<accscalar_t>(max_input);
          simple_reduce(
              item_id, local_acc_max, [](accscalar_t a, accscalar_t b) {
                return Numerics<accscalar_t>::max(a, b);
              });

          // get sum
          auto sum_input = static_cast<accscalar_t>(0);
          for (size_t i = local_id; i < dim_size; i += local_size) {
            // (NOTE) This arithmetic convension is to avoid dp_global_ptr cast
            auto in_value = static_cast<accscalar_t>(in_ptr[i]);
            sum_input +=
                Numerics<accscalar_t>::exp(in_value - local_acc_max[0]);
          }
          local_acc_sum[local_id] = sum_input;
          simple_reduce(
              item_id, local_acc_sum, [](accscalar_t a, accscalar_t b) {
                return a + b;
              });

          Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(
              local_acc_max[0], local_acc_sum[0]);
          for (size_t i = local_id; i < dim_size; i += local_size) {
            out_ptr[i] = epilogue(in_ptr[i]);
          }
        });
  };

  // launch kernel
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename outscalar_t,
    template <typename, typename, typename>
    class Epilogue>
void softmax_fwd_kernel(
    scalar_t* in_data,
    scalar_t* out_data,
    size_t dim_size,
    size_t dim_stride,
    size_t outer_size,
    size_t local_size) {
  RECORD_FUNCTION("softmax_fwd_kernel", {});
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  size_t group_num = (dim_stride + local_size - 1) / local_size;
  DPCPP::range<2> global_range(outer_size, group_num * local_size);
  DPCPP::range<2> local_range(1, local_size);
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        DPCPP::nd_range<2>(global_range, local_range),
        [=](DPCPP::nd_item<2> item_id) {
          size_t local_id = item_id.get_local_id(1);
          auto group_id_bs = item_id.get_group(0);
          auto group_id_plane = item_id.get_group(1);
          auto in_ptr = in_data + group_id_bs * dim_size * dim_stride +
              group_id_plane * local_size;
          auto out_ptr = out_data + group_id_bs * dim_size * dim_stride +
              group_id_plane * local_size;

          if (group_id_plane * local_size + local_id < dim_stride) {
            auto max_input = in_ptr[local_id];
            for (uint32_t i = 1; i < dim_size; ++i) {
              int64_t offset = i * dim_stride + local_id;
              auto in_value = in_ptr[offset];
              max_input = max_input > in_value ? max_input : in_value;
            }

            auto sum_input = static_cast<accscalar_t>(0);
            for (uint32_t i = 0; i < dim_size; ++i) {
              int64_t offset = i * dim_stride + local_id;
              sum_input +=
                  Numerics<accscalar_t>::exp(in_ptr[offset] - max_input);
            }

            Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(
                max_input, sum_input);

            for (size_t i = 0; i < dim_size; ++i) {
              int64_t offset = i * dim_stride + local_id;
              out_ptr[offset] = epilogue(in_ptr[offset]);
            }
          }
        });
  };

  // launch kernel
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

// softmax = exp(x) / sum(exp(x))
// to ensuare the exp(x) in range of [0, 1], we use exp(x - max_x)
// then softmax = exp(x) / (exp(max_x) * sum(exp(x - max_x)))
// all the kernels for softmax fwd follow this flowchar:
// 1. get the max_x value in the target dim
// 2. get the sum(exp(x - max_x)) value in the target dim
// 3. get softmax value for each element
// We have differnet kernels for softmax fwd because we consider different
// cases like what CPU does:
// 1. target dim is the last dim: softmax_lastdim_fwd_kernel
// 2. target dim is not the last dim: softmax_fwd_kernel
template <
    typename scalar_t,
    typename accscalar_t,
    typename outscalar_t,
    template <typename, typename, typename>
    class Epilogue>
void SpatialSoftMaxForward(
    outscalar_t* output,
    scalar_t* input,
    TensorInfo<scalar_t, uint64_t> outer_info,
    size_t outer_size,
    size_t dim_size,
    size_t dim_stride) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  size_t wgroup_size = dpcppMaxWorkGroupSize(dev_id);

  if (dim_stride == 1) {
    // The wgroup_size is decide on the dim_size
    // Since binary-tree reduce is used in softmax_lastdim_fwd_kernel, if the
    // dim_size is small, larger workgroup_size will make computation resource
    // wasting. If dim_size is very large, smaller workgroup_size will make each
    // workitem for_loop many times for internal reduce. Thus, different
    // workgroup_size will be used for different reduce cases.
    if (dim_size < 1024) {
      wgroup_size = 64;
    } else if (dim_size < 2048) {
      wgroup_size = 128;
    } else if (dim_size < 4096) {
      wgroup_size = 256;
    } else if (dim_size < 8192) {
      wgroup_size = 512;
    }
    size_t local_size = std::min(wgroup_size, dim_size);
    softmax_lastdim_fwd_kernel<scalar_t, accscalar_t, outscalar_t, Epilogue>(
        input,
        output,
        outer_info,
        dim_size,
        dim_stride,
        outer_size,
        local_size);
  } else {
    size_t local_size = std::min(wgroup_size, dim_stride);
    softmax_fwd_kernel<scalar_t, accscalar_t, outscalar_t, Epilogue>(
        input, output, dim_size, dim_stride, outer_size, local_size);
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename outscalar_t,
    template <typename, typename, typename>
    class Epilogue>
void softmax_lastdim_bwd_kernel(
    scalar_t* gradInput,
    const outscalar_t* output,
    const outscalar_t* gradOutput,
    TensorInfo<scalar_t, uint64_t> outer_info,
    size_t dim_size,
    size_t dim_stride,
    size_t outer_size,
    size_t local_size) {
  RECORD_FUNCTION("softmax_lastdim_bwd_kernel", {});
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  size_t global_size = outer_size * local_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto gradInput_data = gradInput;
    auto output_data = output;
    auto gradOutput_data = gradOutput;

    // create SLM
    auto local_acc_sum = dpcpp_local_acc_t<accscalar_t>(local_size, cgh);
    cgh.parallel_for(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(global_size), DPCPP::range<1>(local_size)),
        [=](DPCPP::nd_item<1> item_id) {
          size_t local_id = item_id.get_local_id(0);
          auto group_id = item_id.get_group(0);
          auto data_offset =
              IndexToOffset<outscalar_t, uint64_t>::get(group_id, outer_info);
          auto gradInput_ptr = gradInput_data + data_offset;
          auto output_ptr = output_data + data_offset;
          auto gradOutput_ptr = gradOutput_data + data_offset;

          auto thread_sum = static_cast<accscalar_t>(0);
          for (size_t i = local_id; i < dim_size; i += local_size) {
            thread_sum +=
                static_cast<accscalar_t>(gradOutput_ptr[i * dim_stride]);
          }
          local_acc_sum[local_id] = thread_sum;

          simple_reduce(
              item_id, local_acc_sum, [](accscalar_t a, accscalar_t b) {
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
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename outscalar_t,
    template <typename, typename, typename>
    class Epilogue>
void host_softmax_bwd_kernel(
    scalar_t* gradInput,
    const outscalar_t* output,
    const outscalar_t* gradOutput,
    size_t dim_size,
    size_t dim_stride,
    size_t outer_size,
    size_t local_size) {
  RECORD_FUNCTION("host_softmax_bwd_kernel", {});
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  size_t group_num = (dim_stride + local_size - 1) / local_size;
  DPCPP::range<2> global_range(outer_size, group_num * local_size);
  DPCPP::range<2> local_range(1, local_size);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto gradInput_data = gradInput;
    auto output_data = output;
    auto gradOutput_data = gradOutput;

    cgh.parallel_for(
        DPCPP::nd_range<2>(global_range, local_range),
        [=](DPCPP::nd_item<2> item_id) {
          size_t local_id = item_id.get_local_id(1);
          auto group_id_bs = item_id.get_group(0);
          auto group_id_plane = item_id.get_group(1);
          auto gradInput_ptr = gradInput_data +
              group_id_bs * dim_size * dim_stride + group_id_plane * local_size;
          auto output_ptr = output_data + group_id_bs * dim_size * dim_stride +
              group_id_plane * local_size;
          auto gradOutput_ptr = gradOutput_data +
              group_id_bs * dim_size * dim_stride + group_id_plane * local_size;

          if (group_id_plane * local_size + local_id < dim_stride) {
            auto thread_sum = static_cast<accscalar_t>(0);
            for (size_t i = 0; i < dim_size; ++i) {
              size_t offset = i * dim_stride + local_id;
              thread_sum += static_cast<accscalar_t>(
                  gradOutput_ptr[i * dim_stride + local_id]);
            }

            Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(thread_sum);
            for (size_t i = 0; i < dim_size; ++i) {
              size_t offset = i * dim_stride + local_id;
              gradInput_ptr[offset] =
                  epilogue(gradOutput_ptr[offset], output_ptr[offset]);
            }
          }
        });
  };

  // launch kernel
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

// The same as forward path, we implemented two softmax bwd kernel to handle
// different cases.
template <
    typename scalar_t,
    typename accscalar_t,
    typename outscalar_t,
    template <typename, typename, typename>
    class Epilogue>
void SpatialSoftMaxBackward(
    scalar_t* gradInput,
    const outscalar_t* output,
    const outscalar_t* gradOutput,
    TensorInfo<scalar_t, uint64_t> outer_info,
    size_t outer_size,
    size_t dim_size,
    size_t dim_stride) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  size_t wgroup_size = dpcppMaxWorkGroupSize(dev_id);

  if (dim_stride == 1) {
    // The same situation as forward kernel
    if (dim_size < 1024) {
      wgroup_size = 64;
    } else if (dim_size < 2048) {
      wgroup_size = 128;
    } else if (dim_size < 4096) {
      wgroup_size = 256;
    } else if (dim_size < 8192) {
      wgroup_size = 512;
    }
    size_t local_size = std::min(wgroup_size, dim_size);
    softmax_lastdim_bwd_kernel<scalar_t, accscalar_t, outscalar_t, Epilogue>(
        gradInput,
        output,
        gradOutput,
        outer_info,
        dim_size,
        dim_stride,
        outer_size,
        local_size);
  } else {
    size_t local_size = std::min(wgroup_size, dim_stride);
    host_softmax_bwd_kernel<scalar_t, accscalar_t, outscalar_t, Epilogue>(
        gradInput,
        output,
        gradOutput,
        dim_size,
        dim_stride,
        outer_size,
        local_size);
  }
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
  TORCH_CHECK(
      input_.is_contiguous(),
      "** host_softmax only supports contiguous input tensor");

  Tensor output = at::native::empty_like(input_);
  Tensor input = input_;
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
          size_t outer_numel = input.numel() / (dim_stride * dim_size);
          TensorInfo<scalar_t, uint64_t> outer_info =
              getTensorInfo<scalar_t, uint64_t>(input);
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
  TORCH_CHECK(
      grad_.is_contiguous(),
      "** host_softmax_backward only supports contiguous grad tensor");
  TORCH_CHECK(
      output_.is_contiguous(),
      "** host_softmax_backward only supports contiguous output tensor");

  int64_t dim = maybe_wrap_dim(dim_, grad_.dim());
  Tensor gI = at::empty_like(grad_);

  if (output_.numel() == 0) {
    return gI;
  }

  Tensor grad = grad_;
  if (grad.dim() == 0)
    grad = grad.view(1);
  TORCH_CHECK(
      dim >= 0 && dim < grad.dim(),
      "dim must be non-negative and less than input dimensions");
  Tensor output = output_;
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
        auto outer_numel = output.numel() / (dim_size * dim_stride);
        TensorInfo<outscalar_t, uint64_t> outer_info =
            getTensorInfo<outscalar_t, uint64_t>(output);
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

// We now use DPCPP softmax fwd kernel instead of oneDNN softmax fwd kernel
Tensor _softmax(
    const Tensor& input_,
    const int64_t dim,
    const bool half_to_float) {
  checkBackend("_softmax", {input_}, Backend::XPU);

  // 1.check oneDNN support input scalar_t or not
  // 2.check the input tensor is contiguous or not
  // when satify the aformentioned two conditions,
  // the oneDNN path will be selected,
  // all the other cases will go to DPCPP path
  if (xpu::oneDNN::softmax_valid(input_)) {
    return xpu::oneDNN::softmax(input_, dim, half_to_float);
  } else {
    Tensor input = to_plain_if_needed(input_).contiguous();
    return host_softmax<impl::SoftMaxForwardEpilogue>(
        input, dim, half_to_float);
  }
}

Tensor _softmax_backward_data(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim,
    const Tensor& input) {
  bool half_to_float = grad_.scalar_type() != input.scalar_type();
  if (half_to_float) {
    TORCH_CHECK(
        !half_to_float,
        "softmax backward with half to float "
        "conversion is not supported on DPCPP");
  }

  // 1.check oneDNN support input scalar_t or not
  // 2.check the input tensor is contiguous or not
  // when satify the aformentioned two conditions,
  // the oneDNN path will be selected,
  // all the other cases will go to DPCPP path
  if (xpu::oneDNN::softmax_backward_valid(grad_, output_, input)) {
    return xpu::oneDNN::softmax_backward(grad_, output_, dim, half_to_float);
  } else {
    auto grad = to_plain_if_needed(grad_).contiguous();
    auto output = to_plain_if_needed(output_).contiguous();
    Tensor tmp = grad * output;
    return host_softmax_backward<impl::SoftMaxBackwardEpilogue>(
        tmp, output, dim, half_to_float);
  }
}

Tensor _log_softmax(const Tensor& self_, int64_t dim, bool half_to_float) {
  Tensor self = self_.contiguous();
  return host_softmax<impl::LogSoftMaxForwardEpilogue>(
      self, dim, half_to_float);
}

Tensor _log_softmax_backward_data(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim,
    const Tensor& input) {
  bool half_to_float = grad_.scalar_type() != input.scalar_type();
  if (half_to_float) {
    TORCH_INTERNAL_ASSERT(
        !half_to_float,
        "softmax with half to float conversion is not supported on DPCPP");
  }

  auto grad = grad_.contiguous();
  auto output = output_.contiguous();
  return host_softmax_backward<impl::LogSoftMaxBackwardEpilogue>(
      grad, output, dim, half_to_float);
}

} // namespace AtenIpexTypeXPU
} // namespace at
