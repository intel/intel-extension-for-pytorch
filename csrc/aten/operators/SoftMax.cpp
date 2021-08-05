#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <intrinsic/ipex_intrinsic.h>
#include <core/Memory.h>
#include <core/detail/TensorInfo.h>
#include <runtime/Utils.h>
#include <oneDNN/oneDNN.h>

#include "comm/AccumulateType.h"
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/SimpelReduce.h"

using namespace dnnl;
using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {
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
    TensorInfo<scalar_t, uint64_t> outer_info,
    size_t outer_size,
    size_t dim_size,
    size_t dim_stride) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  size_t local_size = dpcppMaxWorkGroupSize(dev_id);
  local_size = std::min(local_size, dim_size);
  size_t global_size = outer_size * local_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = input;
    auto out_data = output;
    auto local_acc_max = dpcpp_local_acc_t<accscalar_t>(local_size, cgh);
    auto local_acc_sum = dpcpp_local_acc_t<accscalar_t>(local_size, cgh);
    cgh.parallel_for<SpatialSoftmaxForwardKernelName<
        scalar_t,
        Epilogue<scalar_t, accscalar_t, outscalar_t>>>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(global_size), DPCPP::range<1>(local_size)),
        [=](DPCPP::nd_item<1> item_id) {
          size_t local_id = item_id.get_local_id(0);
          auto group_id = item_id.get_group(0);
          auto data_offset =
              IndexToOffset<scalar_t, uint64_t>::get(
                  group_id, outer_info);
          auto in_ptr = in_data + data_offset;
          auto out_ptr = out_data+ data_offset;
          // get max
          auto max_input = in_ptr[0];
          for (uint32_t i = local_id; i < dim_size; i += local_size) {
            max_input =
                Numerics<scalar_t>::max(max_input, in_ptr[i * dim_stride]);
          }
          // to accscalar_t
          local_acc_max[local_id] = static_cast<accscalar_t>(max_input);

          simple_reduce(item_id, local_acc_max, [](accscalar_t a, accscalar_t b) {
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

          simple_reduce(item_id, local_acc_sum, [](accscalar_t a, accscalar_t b) {
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
    const outscalar_t* output,
    const outscalar_t* gradOutput,
    TensorInfo<scalar_t, uint64_t> outer_info,
    size_t outer_size,
    size_t dim_size,
    size_t dim_stride) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  // Because of softmax backward using reduce, 512(max) WG size may cause much work-item waste(1/2 + 1/4 + 1/8) in tree reduction and is harmful to occupancy.
  // Thus, 64 is chosen here to set as WG size. In addition, SLM has 65 banks in ATS and 64 here is extremely avoidable for bank conflict.
  // 64 is both compatible for Gen9 and Gen12.
  size_t local_size = 64;
  local_size = std::min(local_size, dim_size);
  size_t global_size = outer_size * local_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto gradInput_data = gradInput;
    auto output_data = output;
    auto gradOutput_data = gradOutput;

    // create SLM
    auto local_acc_sum = dpcpp_local_acc_t<accscalar_t>(local_size, cgh);
    cgh.parallel_for<SpatialSoftmaxBackwardKernelName<
        scalar_t,
        Epilogue<scalar_t, accscalar_t, outscalar_t>>>(
        DPCPP::nd_range<1>(
            DPCPP::range<1>(global_size), DPCPP::range<1>(local_size)),
        [=](DPCPP::nd_item<1> item_id) {
          size_t local_id = item_id.get_local_id(0);
          auto group_id = item_id.get_group(0);
          auto data_offset =
              IndexToOffset<outscalar_t, uint64_t>::get(
                  group_id, outer_info);
          auto gradInput_ptr = gradInput_data + data_offset;
          auto output_ptr = output_data + data_offset;
          auto gradOutput_ptr = gradOutput_data + data_offset;

          auto thread_sum = static_cast<accscalar_t>(0);
          for (size_t i = local_id; i < dim_size; i += local_size) {
            thread_sum +=
                static_cast<accscalar_t>(gradOutput_ptr[i * dim_stride]);
          }
          local_acc_sum[local_id] = thread_sum;

          simple_reduce(item_id, local_acc_sum, [](accscalar_t a, accscalar_t b) {
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
  RECORD_FUNCTION("host_softmax_backward", std::vector<c10::IValue>({grad_, output_}));
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

void get_dnnl_format(
    const Tensor& input,
    memory::format_tag& dnnl_format,
    memory::dims& input_tz) {

  auto input_sizes = input.sizes();
  auto input_ndim = input_sizes.size();

  if (input_ndim == 1) {
    dnnl_format = memory::format_tag::x;
    input_tz = {input.size(0)};
  } else if (input_ndim == 2) {
    dnnl_format = memory::format_tag::nc;
    input_tz = {input.size(0), input.size(1)};
  } else if (input_ndim == 3) {
    dnnl_format = memory::format_tag::tnc;
    input_tz = {input.size(0), input.size(1), input.size(2)};
  } else if (input_ndim == 4) {
    dnnl_format = memory::format_tag::nchw;
    input_tz = {/*n*/ input.size(0),
                /*c*/ input.size(1),
                /*h*/ input.size(2),
                /*w*/ input.size(3)};
  } else {
    std::stringstream ss;
    ss << "DPCPP softmax backend got shape=" << input_sizes
       << ", expected input with rank 1 to  rank 4 shape";
    AT_ERROR(ss.str());
  }
}

Tensor _softmax_onednn(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float) {

  TORCH_CHECK(input.dim() <= 4 && input.dim() >=1, "Input Dims out of range");

  Device curDevice = Device(at::kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  memory::format_tag dnnl_format;
  memory::dims input_tz;
  get_dnnl_format(input, dnnl_format, input_tz);

  auto data_t = get_onednn_dtype(input);

  auto input_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(input);
  auto input_md = input_ctx.is_plain()? memory::desc({input_tz}, data_t, dnnl_format) :
      input_ctx.meta();

  auto axis = dim < 0 ? dim + input.dim(): dim;

  // Create operation descriptor.
  auto softmax_forward_desc = softmax_forward::desc(prop_kind::forward, input_md, axis);

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key (key, input_md, axis);
#endif

  // Create primitive descriptor.
  auto softmax_forward_pd = softmax_forward::primitive_desc(softmax_forward_desc, engine);

  Tensor output;
  if (input_ctx.is_plain()) {
    output = at::empty_like(input);
  } else {
    output = empty_opaque_tensor(softmax_forward_pd.dst_desc(), input.options(), c10::nullopt);
  }
  auto input_usr_memory = dpcpp_onednn_memory(input_md, engine, input.data_ptr());
  auto output_usr_memory = dpcpp_onednn_memory(softmax_forward_pd.dst_desc(), engine, output.data_ptr());

  // Create the primitive.
#ifdef USE_PRIMITIVE_CACHE
  auto softmax_onednn_forward =
      fetch_or_create_m<softmax_forward>(key, softmax_forward_pd);
#else
  auto softmax_onednn_forward = softmax_forward(softmax_forward_pd);
#endif

  // Primitive execution.
  DPCPP_ONEDNN_EXEC(
      softmax_onednn_forward,
      strm,
      {{DNNL_ARG_SRC, input_usr_memory},
       {DNNL_ARG_DST, output_usr_memory}});

  return output;
}

Tensor _softmax_backward_onednn(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    bool half_to_float) {

  TORCH_CHECK(grad.dim() <= 4 && grad.dim() >=1, "Input Dims out of range");

  Device curDevice = Device(at::kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();
  auto gI = at::empty_like(grad);
  memory::format_tag output_dnnl_format;
  memory::format_tag grad_dnnl_format;
  memory::dims output_tz;
  memory::dims grad_tz;

  get_dnnl_format(output, output_dnnl_format, output_tz);
  get_dnnl_format(grad, grad_dnnl_format, grad_tz);

  auto output_t = get_onednn_dtype(output);
  auto grad_t = get_onednn_dtype(grad);

  auto axis = dim < 0 ? dim + grad.dim(): dim;

  auto output_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(output);
  auto output_md = output_ctx.is_plain() ? memory::desc({output_tz}, output_t, output_dnnl_format) :
      output_ctx.meta();
  auto output_memory = dpcpp_onednn_memory(output_md, engine, output.data_ptr());

  auto grad_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(grad);
  auto grad_md = grad_ctx.is_plain() ? memory::desc({grad_tz, grad_t, grad_dnnl_format}) :
      grad_ctx.meta();
  auto grad_usr_memory = dpcpp_onednn_memory(grad_md, engine, grad.data_ptr());

  auto softmax_forward_desc = softmax_forward::desc(prop_kind::forward, output_md, axis);
  auto softmax_forward_pd = softmax_forward::primitive_desc(softmax_forward_desc, engine);

  Tensor grad_opt;
  auto grad_memory = grad_usr_memory;
  if (grad_ctx.is_plain() && (!output_ctx.is_plain())) {
    grad_opt = empty_opaque_tensor(softmax_forward_pd.dst_desc(), grad.options(), c10::nullopt);
    grad_memory = dpcpp_onednn_memory(softmax_forward_pd.dst_desc(), engine, grad_opt.data_ptr());
    grad_md = softmax_forward_pd.dst_desc();
    xpu::oneDNN::reorder(grad, grad_opt);
  }

  auto softmax_backward_desc = softmax_backward::desc(grad_md, output_md, axis);
  auto softmax_backward_pd = softmax_backward::primitive_desc(
                                                 softmax_backward_desc,
                                                 engine,
                                                 softmax_forward_pd);
#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key (key, grad_md, output_md, axis);
#endif

  auto plain_gi_md = memory::desc({grad_tz, grad_t, grad_dnnl_format});
  auto expected_gi_md = softmax_backward_pd.diff_src_desc();
  if (plain_gi_md != expected_gi_md) {
    gI = empty_opaque_tensor(expected_gi_md, grad.options(), c10::nullopt);
  }
  auto gi_memory = dpcpp_onednn_memory(expected_gi_md, engine, gI.data_ptr());

  // Create the primitive.
#ifdef USE_PRIMITIVE_CACHE
  auto softmax_onednn_backward =
      fetch_or_create_m<softmax_backward>(key, softmax_backward_pd);
#else
  auto softmax_onednn_backward = softmax_backward(softmax_backward_pd);
#endif

  // Primitive execution.
  DPCPP_ONEDNN_EXEC(
      softmax_onednn_backward,
      strm,
      {{DNNL_ARG_DST, output_memory},
      {DNNL_ARG_DIFF_SRC, gi_memory},
      {DNNL_ARG_DIFF_DST, grad_memory}});

  return gI;
}

Tensor _softmax(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float) {

  checkBackend("_softmax", {input}, Backend::XPU);

  if (input.scalar_type() != at::ScalarType::Float &&
      input.scalar_type() != at::ScalarType::Half &&
      input.scalar_type() != at::ScalarType::BFloat16) {
    // oneDNN does not support this type. Use native kernel instead.
    return host_softmax<impl::SoftMaxForwardEpilogue>(input, dim, half_to_float);
  } else {
    return _softmax_onednn(input, dim, half_to_float);
  }
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
  // Skip oneDNN kernel due to its bad performance. 
  /*if ((input.scalar_type() == at::ScalarType::Float ||
      input.scalar_type() == at::ScalarType::BFloat16) &&
      grad.is_contiguous() && output.is_contiguous()) {
    return _softmax_backward_onednn(grad, output, dim, half_to_float);
  } else {*/
    // Use native kernel instead.
    Tensor tmp = grad * output;
    return host_softmax_backward<impl::SoftMaxBackwardEpilogue>(
      tmp, output, dim, half_to_float);

  //}
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

} // namespace AtenIpexTypeXPU
} // namespace at
