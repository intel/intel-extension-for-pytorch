#pragma once

#include <core/Runtime.h>
#include <dnnl.hpp>

using namespace mkldnn;
using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

class scalar_t_to_dnnl {
 public:
  const memory::data_type data_t;
  template <
      typename scalar_t,
      c10::guts::enable_if_t<std::is_same<scalar_t, Half>::value, int> = 0>
  static memory::data_type to() {
    return memory::data_type::f16;
  };

  template <
      typename scalar_t,
      c10::guts::enable_if_t<std::is_same<scalar_t, BFloat16>::value, int> = 0>
  static memory::data_type to() {
    return memory::data_type::bf16;
  };

  template <
      typename scalar_t,
      c10::guts::enable_if_t<std::is_same<scalar_t, float>::value, int> = 0>
  static memory::data_type to() {
    return memory::data_type::f32;
  };

  template <
      typename scalar_t,
      c10::guts::enable_if_t<std::is_same<scalar_t, double>::value, int> = 0>
  static memory::data_type to() {
    TORCH_CHECK(0, " mkldnn not support for double");
  };
};

template <typename scalar_t>
static void avg_pool_out_frame(
    scalar_t* input_data,
    scalar_t* output_data,
    int64_t nbatch,
    int64_t nInputPlane,
    int64_t inputDepth,
    int64_t inputHeight,
    int64_t inputWidth,
    int64_t outputDepth,
    int64_t outputHeight,
    int64_t outputWidth,
    int kD,
    int kH,
    int kW,
    int dD,
    int dH,
    int dW,
    int padD,
    int padH,
    int padW,
    algorithm alg_kind,
    prop_kind prop_kind) {
  Device curDevice = Device(kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  auto data_t = scalar_t_to_dnnl::to<scalar_t>();
  if (data_t == memory::data_type::f16) {
    prop_kind = dnnl::prop_kind::forward_inference;
  }

  memory::format_tag format;

  memory::dims input_tz;
  memory::dims output_tz;
  memory::dims kernel;
  memory::dims stride;
  memory::dims padding;

  if (inputDepth == 0) {
    format = memory::format_tag::nchw;

    input_tz = {nbatch, nInputPlane, inputHeight, inputWidth};
    output_tz = {nbatch, nInputPlane, outputHeight, outputWidth};
    kernel = {kH, kW};
    stride = {dH, dW};
    padding = {padH, padW};
  } else {
    format = memory::format_tag::ncdhw;

    input_tz = {nbatch, nInputPlane, inputDepth, inputHeight, inputWidth};
    output_tz = {nbatch, nInputPlane, outputDepth, outputHeight, outputWidth};
    kernel = {kD, kH, kW};
    stride = {dD, dH, dW};
    padding = {padD, padH, padW};
  }

  auto input_md = memory::desc({input_tz}, data_t, format);
  auto output_md = memory::desc({output_tz}, data_t, format);

  auto input_usr_memory = memory({{{input_tz}, data_t, format}, engine});
  dpcpp_set_mkldnn_buffer(input_data, input_usr_memory);

  auto output_usr_memory = memory({{{output_tz}, data_t, format}, engine});
  dpcpp_set_mkldnn_buffer(output_data, output_usr_memory);

  std::shared_ptr<pooling_forward::desc> pooling_forward_desc;
  pooling_forward_desc.reset(new pooling_forward::desc(
      prop_kind,
      alg_kind,
      input_md,
      output_md,
      stride,
      kernel,
      padding,
      padding));

  std::shared_ptr<pooling_forward::primitive_desc> pooling_forward_pd;
  pooling_forward_pd.reset(
      new pooling_forward::primitive_desc(*pooling_forward_desc, engine));

  // auto input_d = pooling_forward_pd->src_desc();
  auto input_memory = input_usr_memory;

  // Currently, SYCL path doesn't support internal format.
  // input has the same format with input_usr.
  // if (input_usr_memory.get_desc() != input_d) {
  //   input_memory = memory(input_d, engine);
  //   reorder(input_usr_memory, input_memory).
  //       execute(strm, input_usr_memory, input_memory);
  // }

  // auto output_d = pooling_forward_pd->dst_desc();
  auto output_memory = output_usr_memory;

  // output has the same format with output_usr.
  // if (output_usr_memory.get_desc() != output_d) {
  //   output_memory = memory(output_d, engine);
  // }

  std::shared_ptr<pooling_forward> pool_forward;
  pool_forward.reset(new pooling_forward(*pooling_forward_pd));
  pool_forward->execute(
      strm, {{MKLDNN_ARG_SRC, input_memory}, {MKLDNN_ARG_DST, output_memory}});

  // reorder output
  // if (output_memory != output_usr_memory) {
  //   reorder(output_memory, output_usr_memory).
  //       execute(strm, output_memory, output_usr_memory);
  // }
}

template <typename scalar_t>
static void avg_pool_backward_out_frame(
    scalar_t* gradInput_data,
    scalar_t* gradOutput_data,
    int64_t nbatch,
    int64_t nInputPlane,
    int64_t inputDepth,
    int64_t inputHeight,
    int64_t inputWidth,
    int64_t outputDepth,
    int64_t outputHeight,
    int64_t outputWidth,
    int kD,
    int kH,
    int kW,
    int dD,
    int dH,
    int dW,
    int padD,
    int padH,
    int padW,
    algorithm alg_kind,
    prop_kind prop_kind) {
  at::Device curDevice = at::Device(kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  auto data_t = scalar_t_to_dnnl::to<scalar_t>();
  if (data_t == memory::data_type::f16) {
    // rise error
  }

  memory::format_tag format;

  memory::dims input_tz;
  memory::dims output_tz;
  memory::dims kernel;
  memory::dims stride;
  memory::dims padding;

  if (inputDepth == 0) {
    format = memory::format_tag::nchw;

    input_tz = {nbatch, nInputPlane, inputHeight, inputWidth};
    output_tz = {nbatch, nInputPlane, outputHeight, outputWidth};
    kernel = {kH, kW};
    stride = {dH, dW};
    padding = {padH, padW};
  } else {
    format = memory::format_tag::ncdhw;

    input_tz = {nbatch, nInputPlane, inputDepth, inputHeight, inputWidth};
    output_tz = {nbatch, nInputPlane, outputDepth, outputHeight, outputWidth};
    kernel = {kD, kH, kW};
    stride = {dD, dH, dW};
    padding = {padD, padH, padW};
  }

  auto input_md = memory::desc({input_tz}, data_t, format);
  auto output_md = memory::desc({output_tz}, data_t, format);

  auto diff_dst_usr_memory = memory({{{output_tz}, data_t, format}, engine});
  dpcpp_set_mkldnn_buffer(gradOutput_data, diff_dst_usr_memory);

  auto diff_src_usr_memory = memory({{{input_tz}, data_t, format}, engine});
  dpcpp_set_mkldnn_buffer(gradInput_data, diff_src_usr_memory);

  std::shared_ptr<pooling_forward::desc> pooling_forward_desc;
  pooling_forward_desc.reset(new pooling_forward::desc(
      prop_kind,
      alg_kind,
      input_md,
      output_md,
      stride,
      kernel,
      padding,
      padding));
  std::shared_ptr<pooling_forward::primitive_desc> pooling_forward_pd;
  pooling_forward_pd.reset(
      new pooling_forward::primitive_desc(*pooling_forward_desc, engine));

  std::shared_ptr<pooling_backward::desc> pooling_backward_desc;
  pooling_backward_desc.reset(new pooling_backward::desc(
      alg_kind, input_md, output_md, stride, kernel, padding, padding));
  std::shared_ptr<pooling_backward::primitive_desc> pooling_backward_pd;
  pooling_backward_pd.reset(new pooling_backward::primitive_desc(
      *pooling_backward_desc, engine, *pooling_forward_pd));

  // auto diff_dst_md = pooling_backward_pd->diff_dst_desc();
  auto diff_dst_memory = diff_dst_usr_memory;

  // Currently, SYCL path doesn't support internal format.
  // diff_dst has the same format with dst.
  // if (diff_dst_usr_memory.get_desc() != diff_dst_md) {
  //   diff_dst_memory = memory(diff_dst_md, engine);
  //   reorder(diff_dst_usr_memory, diff_dst_memory).
  //       execute(strm, diff_dst_usr_memory, diff_dst_memory);
  // }

  // auto diff_src_md = pooling_backward_pd->diff_src_desc();
  auto diff_src_memory = diff_src_usr_memory;

  // diff_src has the same format with src.
  // if (diff_src_usr_memory.get_desc() != diff_src_md) {
  //   diff_src_memory = memory(diff_src_md, engine);
  // }

  std::shared_ptr<pooling_backward> pool_backward;
  pool_backward.reset(new pooling_backward(*pooling_backward_pd));

  pool_backward->execute(
      strm,
      {{MKLDNN_ARG_DIFF_DST, diff_dst_memory},
       {MKLDNN_ARG_DIFF_SRC, diff_src_memory}});

  // Reorder diff_src
  // if (diff_src_memory != diff_src_usr_memory) {
  //   reorder(diff_src_memory, diff_src_usr_memory).
  //       execute(strm, diff_src_memory, diff_src_usr_memory);
  // }
}

} // namespace impl
} // namespace AtenIpexTypeDPCPP
} // namespace at