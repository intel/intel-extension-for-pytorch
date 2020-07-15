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
      std::enable_if_t<std::is_same<scalar_t, Half>::value, int> = 0>
  static memory::data_type to() {
    return memory::data_type::f16;
  };

  template <
      typename scalar_t,
      std::enable_if_t<std::is_same<scalar_t, BFloat16>::value, int> = 0>
  static memory::data_type to() {
    return memory::data_type::bf16;
  };

  template <
      typename scalar_t,
      std::enable_if_t<std::is_same<scalar_t, float>::value, int> = 0>
  static memory::data_type to() {
    return memory::data_type::f32;
  };

  template <
      typename scalar_t,
      std::enable_if_t<std::is_same<scalar_t, double>::value, int> = 0>
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

  auto input_usr_memory = dpcpp_onednn_memory(
      {{input_tz}, data_t, format}, engine, input_data);

  auto output_usr_memory = dpcpp_onednn_memory(
      {{output_tz}, data_t, format}, engine, output_data);

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
  //   DPCPP_ONEDNN_EXEC(reorder(input_usr_memory, input_memory),
  //       strm, input_usr_memory, input_memory);
  // }

  // auto output_d = pooling_forward_pd->dst_desc();
  auto output_memory = output_usr_memory;

  // output has the same format with output_usr.
  // if (output_usr_memory.get_desc() != output_d) {
  //   output_memory = memory(output_d, engine);
  // }

  std::shared_ptr<pooling_forward> pool_forward;
  pool_forward.reset(new pooling_forward(*pooling_forward_pd));
  DPCPP_ONEDNN_EXEC(*pool_forward, strm,
    {{MKLDNN_ARG_SRC, input_memory}, {MKLDNN_ARG_DST, output_memory}});

  // reorder output
  // if (output_memory != output_usr_memory) {
  //   DPCPP_ONEDNN_EXEC(reorder(output_memory, output_usr_memory),
  //       strm, output_memory, output_usr_memory);
  // }
}

template <typename scalar_t>
static void avg_pool_backward_out_frame(
    scalar_t* gradInput_data,
    scalar_t* gradOutput_data,
    int64_t nbatch,
    int64_t nInputPlane,
    int64_t gradInputDepth,
    int64_t gradInputHeight,
    int64_t gradInputWidth,
    int64_t gradOutputDepth,
    int64_t gradOutputHeight,
    int64_t gradOutputWidth,
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

  memory::dims gradInput_tz;
  memory::dims gradOutput_tz;
  memory::dims kernel;
  memory::dims stride;
  memory::dims padding;

  if (gradInputDepth == 0) {
    format = memory::format_tag::nchw;

    gradInput_tz = {nbatch, nInputPlane, gradInputHeight, gradInputWidth};
    gradOutput_tz = {nbatch, nInputPlane, gradOutputHeight, gradOutputWidth};
    kernel = {kH, kW};
    stride = {dH, dW};
    padding = {padH, padW};
  } else {
    format = memory::format_tag::ncdhw;

    gradInput_tz = {
        nbatch, nInputPlane, gradInputDepth, gradInputHeight, gradInputWidth};
    gradOutput_tz = {nbatch,
                     nInputPlane,
                     gradOutputDepth,
                     gradOutputHeight,
                     gradOutputWidth};
    kernel = {kD, kH, kW};
    stride = {dD, dH, dW};
    padding = {padD, padH, padW};
  }

  auto gradInput_md = memory::desc({gradInput_tz}, data_t, format);
  auto gradOutput_md = memory::desc({gradOutput_tz}, data_t, format);

  auto diff_dst_usr_memory = dpcpp_onednn_memory(
      {{gradOutput_tz}, data_t, format}, engine, gradOutput_data);

  auto diff_src_usr_memory = dpcpp_onednn_memory(
      {{gradInput_tz}, data_t, format}, engine, gradInput_data);

  std::shared_ptr<pooling_forward::desc> pooling_forward_desc;
  pooling_forward_desc.reset(new pooling_forward::desc(
      prop_kind,
      alg_kind,
      gradInput_md,
      gradOutput_md,
      stride,
      kernel,
      padding,
      padding));
  std::shared_ptr<pooling_forward::primitive_desc> pooling_forward_pd;
  pooling_forward_pd.reset(
      new pooling_forward::primitive_desc(*pooling_forward_desc, engine));

  std::shared_ptr<pooling_backward::desc> pooling_backward_desc;
  pooling_backward_desc.reset(new pooling_backward::desc(
      alg_kind, gradInput_md, gradOutput_md, stride, kernel, padding, padding));
  std::shared_ptr<pooling_backward::primitive_desc> pooling_backward_pd;
  pooling_backward_pd.reset(new pooling_backward::primitive_desc(
      *pooling_backward_desc, engine, *pooling_forward_pd));

  // auto diff_dst_md = pooling_backward_pd->diff_dst_desc();
  auto diff_dst_memory = diff_dst_usr_memory;

  // Currently, SYCL path doesn't support internal format.
  // diff_dst has the same format with dst.
  // if (diff_dst_usr_memory.get_desc() != diff_dst_md) {
  //   diff_dst_memory = memory(diff_dst_md, engine);
  //   DPCPP_ONEDNN_EXEC(reorder(diff_dst_usr_memory, diff_dst_memory),
  //       strm, diff_dst_usr_memory, diff_dst_memory);
  // }

  // auto diff_src_md = pooling_backward_pd->diff_src_desc();
  auto diff_src_memory = diff_src_usr_memory;

  // diff_src has the same format with src.
  // if (diff_src_usr_memory.get_desc() != diff_src_md) {
  //   diff_src_memory = memory(diff_src_md, engine);
  // }

  std::shared_ptr<pooling_backward> pool_backward;
  pool_backward.reset(new pooling_backward(*pooling_backward_pd));

  DPCPP_ONEDNN_EXEC(*pool_backward, strm,
      {{MKLDNN_ARG_DIFF_DST, diff_dst_memory},
       {MKLDNN_ARG_DIFF_SRC, diff_src_memory}});

  // Reorder diff_src
  // if (diff_src_memory != diff_src_usr_memory) {
  //   DPCPP_ONEDNN_EXEC(reorder(diff_src_memory, diff_src_usr_memory),
  //       strm, diff_src_memory, diff_src_usr_memory);
  // }
}

template <typename scalar_t>
static void max_pool_out_frame(
    scalar_t* input_data,
    scalar_t* output_data,
    int64_t* indices_data,
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

  // Currently, MKLDNN GPU doens't support format_any in pooling
  auto input_md = memory::desc({input_tz}, data_t, format);
  auto output_md = memory::desc({output_tz}, data_t, format);

  auto input_usr_memory = dpcpp_onednn_memory(input_md, engine, input_data);

  auto output_usr_memory = dpcpp_onednn_memory(output_md, engine, output_data);

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

  // auto expected_input_md = pooling_forward_pd->src_desc();
  auto input_memory = input_usr_memory;

  // Currently, SYCL path doesn't support internal format.
  // input has the same format with input_usr.
  // if (input_md != expected_input_md) {
  //   input_memory = memory(expected_input_md, engine);
  //   DPCPP_ONEDNN_EXEC(reorder(input_usr_memory, input_memory),
  //       strm, input_usr_memory, input_memory);
  // }

  // auto expected_output_md = pooling_forward_pd->dst_desc();
  auto output_memory = output_usr_memory;

  // output has the same format with output_usr.
  // if (output_md != expected_output_md) {
  //   output_memory = memory(expected_output_md, engine);
  // }

  auto indices_md = pooling_forward_pd->workspace_desc();

  auto indices_usr =
      at::empty({output_tz}, at::TensorOptions(kDPCPP).dtype(kInt));
  auto indices_usr_memory = dpcpp_onednn_memory(
      {{output_tz}, data_t, format}, engine, indices_usr.data_ptr());
  memory indices_memory = indices_usr_memory;

  std::shared_ptr<pooling_forward> pool_forward;
  pool_forward.reset(new pooling_forward(*pooling_forward_pd));

  // indices has the same format with indices_usr.
  // if (indices_usr_memory.get_desc() != indices_md) {
  //   indices_memory = memory(indices_md, engine);
  // }

  DPCPP_ONEDNN_EXEC(*pool_forward, strm,
      {{MKLDNN_ARG_SRC, input_memory},
       {MKLDNN_ARG_DST, output_memory},
       {MKLDNN_ARG_WORKSPACE, indices_memory}});

  // reorder output
  // if (output_memory != output_usr_memory) {
  //   DPCPP_ONEDNN_EXEC(reorder(output_memory, output_usr_memory),
  //       strm, output_memory, output_usr_memory);
  // }

  // reorder workgroup

  // if (indices_usr_memory.get_desc() != indices_md) {
  //   DPCPP_ONEDNN_EXEC(reorder(indices_memory, indices_usr_memory),
  //       strm, indices_memory, indices_usr_memory);
  // }

  // DPCPP_ONEDNN_EXEC(reorder(indices_memory, indices_usr_memory),
  //         strm, indices_memory, indices_usr_memory);

  dpcppMemoryCopyType(
      (int64_t*)indices_data,
      indices_usr.data_ptr<int32_t>(),
      indices_usr.numel());
}

template <typename scalar_t>
static void max_pool_backward_out_frame(
    scalar_t* gradInput_data,
    scalar_t* gradOutput_data,
    int64_t* indices_data,
    int64_t nbatch,
    int64_t nInputPlane,
    int64_t gradInputDepth,
    int64_t gradInputHeight,
    int64_t gradInputWidth,
    int64_t gradOutputDepth,
    int64_t gradOutputHeight,
    int64_t gradOutputWidth,
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
  at::Device curDevice = at::Device(at::kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  // auto data_t = memory::data_type::f32;
  // auto format_nchw = memory::format_tag::nchw;

  // memory::dims gradInput_tz = {nbatch, nPlane, gradInputHeight,
  // gradInputWidth}; memory::dims gradOutput_tz = {
  //     nbatch, nPlane, gradOutputHeight, gradOutputWidth};
  // memory::dims kernel = {kH, kW};
  // memory::dims stride = {dH, dW};
  // memory::dims padding = {padH, padW};

  auto data_t = scalar_t_to_dnnl::to<scalar_t>();
  if (data_t == memory::data_type::f16) {
    // rise error
  }

  memory::format_tag format;
  memory::dims gradInput_tz;
  memory::dims gradOutput_tz;
  memory::dims kernel;
  memory::dims stride;
  memory::dims padding;

  if (gradInputDepth == 0) {
    format = memory::format_tag::nchw;
    gradInput_tz = {nbatch, nInputPlane, gradInputHeight, gradInputWidth};
    gradOutput_tz = {nbatch, nInputPlane, gradOutputHeight, gradOutputWidth};
    kernel = {kH, kW};
    stride = {dH, dW};
    padding = {padH, padW};
  } else {
    format = memory::format_tag::ncdhw;
    gradInput_tz = {
        nbatch, nInputPlane, gradInputDepth, gradInputHeight, gradInputWidth};
    gradOutput_tz = {nbatch,
                     nInputPlane,
                     gradOutputDepth,
                     gradOutputHeight,
                     gradOutputWidth};
    kernel = {kD, kH, kW};
    stride = {dD, dH, dW};
    padding = {padD, padH, padW};
  }

  // Currently, MKLDNN GPU doens't support format_any in pooling
  auto gradInput_md = memory::desc({gradInput_tz}, data_t, format);
  auto gradOutput_md = memory::desc({gradOutput_tz}, data_t, format);

  auto diff_dst_usr_memory = dpcpp_onednn_memory(
      {{gradOutput_tz}, data_t, format}, engine, gradOutput_data);

  auto diff_src_usr_memory = dpcpp_onednn_memory(
      {{gradInput_tz}, data_t, format}, engine, gradInput_data);

  std::shared_ptr<pooling_forward::desc> pooling_forward_desc;
  pooling_forward_desc.reset(new pooling_forward::desc(
      prop_kind,
      alg_kind,
      gradInput_md,
      gradOutput_md,
      stride,
      kernel,
      padding,
      padding));
  std::shared_ptr<pooling_forward::primitive_desc> pooling_forward_pd;
  pooling_forward_pd.reset(
      new pooling_forward::primitive_desc(*pooling_forward_desc, engine));

  std::shared_ptr<pooling_backward::desc> pooling_backward_desc;
  pooling_backward_desc.reset(new pooling_backward::desc(
      alg_kind, gradInput_md, gradOutput_md, stride, kernel, padding, padding));
  std::shared_ptr<pooling_backward::primitive_desc> pooling_backward_pd;
  pooling_backward_pd.reset(new pooling_backward::primitive_desc(
      *pooling_backward_desc, engine, *pooling_forward_pd));

  // auto diff_dst_md = pooling_backward_pd->diff_dst_desc();
  auto diff_dst_memory = diff_dst_usr_memory;

  // Currently, SYCL path doesn't support internal format.
  // diff_dst has the same format with dst.
  // if (diff_dst_usr_memory.get_desc() != diff_dst_md) {
  //   diff_dst_memory = memory(diff_dst_md, engine);
  //   DPCPP_ONEDNN_EXEC(reorder(diff_dst_usr_memory, diff_dst_memory),
  //       strm, diff_dst_usr_memory, diff_dst_memory);
  // }

  // auto expected_diff_src_pd = pooling_backward_pd->diff_src_desc();
  auto diff_src_memory = diff_src_usr_memory;

  // diff_src has the same format with src.
  // if (diff_src_usr_memory.get_desc() != expected_diff_src_pd) {
  //   diff_src_memory = memory(expected_diff_src_pd, engine);
  // }

  std::shared_ptr<pooling_backward> pool_backward;

  auto indices_usr =
      at::empty({gradOutput_tz}, at::TensorOptions(kDPCPP).dtype(kInt));
  dpcppMemoryCopyType(
      indices_usr.data_ptr<int32_t>(),
      (int64_t*)indices_data,
      indices_usr.numel());

  pool_backward.reset(new pooling_backward(*pooling_backward_pd));

  auto indices_md = pooling_forward_pd->workspace_desc();
  auto indices_usr_memory = dpcpp_onednn_memory(
      {{gradOutput_tz}, (memory::data_type)indices_md.data.data_type, format},
       engine, indices_usr.data_ptr());
  auto indices_memory = indices_usr_memory;

  // indices has the same format with indices.
  // reorder indices if needed
  // if (indices_usr_memory.get_desc() != indices_md) {
  //   DPCPP_ONEDNN_EXEC(reorder(indices_usr_memory, indices_memory),
  //       strm, indices_usr_memory, indices_memory);
  // }
  DPCPP_ONEDNN_EXEC(*pool_backward, strm,
      {{MKLDNN_ARG_DIFF_DST, diff_dst_memory},
       {MKLDNN_ARG_DIFF_SRC, diff_src_memory},
       {MKLDNN_ARG_WORKSPACE, indices_memory}});

  // Reorder diff_src
  // if (diff_src_memory != diff_src_usr_memory) {
  //   DPCPP_ONEDNN_EXEC(reorder(diff_src_memory, diff_src_usr_memory),
  //       strm, diff_src_memory, diff_src_usr_memory);
  // }
}

} // namespace impl
} // namespace AtenIpexTypeDPCPP
} // namespace at
