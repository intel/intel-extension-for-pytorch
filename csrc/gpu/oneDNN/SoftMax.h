#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <oneDNN/Runtime.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <utils/LRUCache.h>
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace at::AtenIpexTypeXPU;

namespace xpu {
namespace oneDNN {

static inline void get_dnnl_format(
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
    input_tz = {
        /*n*/ input.size(0),
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

static Tensor softmax(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float,
    Tensor& output) {
  TORCH_CHECK(input.dim() <= 4 && input.dim() >= 1, "Input Dims out of range");

  Device curDevice = Device(at::kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  memory::format_tag dnnl_format;
  memory::dims input_tz;
  get_dnnl_format(input, dnnl_format, input_tz);

  auto data_t = get_onednn_dtype(input);

  auto input_ctx =
      at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(input);
  auto input_md = input_ctx.is_plain()
      ? memory::desc({input_tz}, data_t, get_onednn_strides(input))
      : input_ctx.meta();

  auto axis = dim < 0 ? dim + input.dim() : dim;

  // Create operation descriptor.
  auto softmax_forward_desc =
      dnnl::softmax_forward::desc(prop_kind::forward, input_md, axis);

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, input_md, axis);
#endif

  // Create primitive descriptor.
  auto softmax_forward_pd =
      dnnl::softmax_forward::primitive_desc(softmax_forward_desc, engine);

  if (!output.defined()) {
    if (input_ctx.is_plain()) {
      output = at::empty_like(input);
    } else {
      output = empty_opaque_tensor(
          softmax_forward_pd.dst_desc(), input.options(), c10::nullopt);
    }
  }
  auto input_usr_memory =
      dpcpp_onednn_memory(input_md, engine, input.data_ptr());
  auto output_usr_memory = dpcpp_onednn_memory(
      softmax_forward_pd.dst_desc(), engine, output.data_ptr());

  // Create the primitive.
#ifdef USE_PRIMITIVE_CACHE
  auto softmax_onednn_forward =
      fetch_or_create_m<dnnl::softmax_forward>(key, softmax_forward_pd);
#else
  auto softmax_onednn_forward = dnnl::softmax_forward(softmax_forward_pd);
#endif

  // Primitive execution.
  DPCPP_ONEDNN_EXEC(
      softmax_onednn_forward,
      strm,
      {{DNNL_ARG_SRC, input_usr_memory}, {DNNL_ARG_DST, output_usr_memory}});

  return output;
}

static Tensor softmax_backward(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    bool half_to_float,
    Tensor gI) {
  TORCH_CHECK(grad.dim() <= 4 && grad.dim() >= 1, "Input Dims out of range");

  Device curDevice = Device(at::kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();
  if (!gI.defined()) {
    gI = at::empty_like(grad);
  }
  memory::format_tag output_dnnl_format;
  memory::format_tag grad_dnnl_format;
  memory::dims output_tz;
  memory::dims grad_tz;

  get_dnnl_format(output, output_dnnl_format, output_tz);
  get_dnnl_format(grad, grad_dnnl_format, grad_tz);

  auto output_t = get_onednn_dtype(output);
  auto grad_t = get_onednn_dtype(grad);

  auto axis = dim < 0 ? dim + grad.dim() : dim;

  auto output_ctx =
      at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(output);
  auto output_md = output_ctx.is_plain()
      ? memory::desc({output_tz}, output_t, output_dnnl_format)
      : output_ctx.meta();
  auto output_memory =
      dpcpp_onednn_memory(output_md, engine, output.data_ptr());

  auto grad_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(grad);
  auto grad_md = grad_ctx.is_plain()
      ? memory::desc({grad_tz, grad_t, grad_dnnl_format})
      : grad_ctx.meta();
  auto grad_usr_memory = dpcpp_onednn_memory(grad_md, engine, grad.data_ptr());

  auto softmax_forward_desc =
      softmax_forward::desc(prop_kind::forward, output_md, axis);
  auto softmax_forward_pd =
      softmax_forward::primitive_desc(softmax_forward_desc, engine);

  Tensor grad_opt;
  auto grad_memory = grad_usr_memory;
  if (grad_ctx.is_plain() && (!output_ctx.is_plain())) {
    grad_opt = empty_opaque_tensor(
        softmax_forward_pd.dst_desc(), grad.options(), c10::nullopt);
    grad_memory = dpcpp_onednn_memory(
        softmax_forward_pd.dst_desc(), engine, grad_opt.data_ptr());
    grad_md = softmax_forward_pd.dst_desc();
    xpu::oneDNN::reorder(grad, grad_opt);
  }

  auto softmax_backward_desc =
      dnnl::softmax_backward::desc(grad_md, output_md, axis);
  auto softmax_backward_pd = dnnl::softmax_backward::primitive_desc(
      softmax_backward_desc, engine, softmax_forward_pd);
#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, grad_md, output_md, axis);
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
      fetch_or_create_m<dnnl::softmax_backward>(key, softmax_backward_pd);
#else
  auto softmax_onednn_backward = dnnl::softmax_backward(softmax_backward_pd);
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

} // namespace oneDNN
} // namespace xpu
