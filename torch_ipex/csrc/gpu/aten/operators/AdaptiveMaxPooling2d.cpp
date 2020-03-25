#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <core/Runtime.h>
#include <vector>

using namespace dnnl;
using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
static void adaptive_max_pool2d_out_frame(
    scalar_t* input_data,
    scalar_t* output_data,
    int64_t* indices_data,
    int64_t nbatch,
    int64_t nInputPlane,
    int64_t inputWidth,
    int64_t inputHeight,
    int64_t outputWidth,
    int64_t outputHeight,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    algorithm alg_kind,
    prop_kind prop_kind) {
  at::Device curDevice = at::Device(kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  auto data_t = memory::data_type::f32;
  if (std::is_same<scalar_t, Half>::value == true) {
    data_t = memory::data_type::f16;
    prop_kind = dnnl::prop_kind::forward_inference;
  }
  auto format_nchw = memory::format_tag::nchw;

  memory::dims input_tz = {nbatch, nInputPlane, inputHeight, inputWidth};
  memory::dims output_tz = {nbatch, nInputPlane, outputHeight, outputWidth};
  memory::dims kernel = {kH, kW};
  memory::dims stride = {dH, dW};
  memory::dims padding = {padH, padW};

  // Currently, MKLDNN GPU doens't support format_any in pooling
  auto input_md = memory::desc({input_tz}, data_t, format_nchw);
  auto output_md = memory::desc({output_tz}, data_t, format_nchw);

  auto input_usr_memory = memory(input_md, engine);
  dpcpp_set_mkldnn_buffer(input_data, input_usr_memory);

  auto output_usr_memory = memory(output_md, engine);
  dpcpp_set_mkldnn_buffer(output_data, output_usr_memory);

  std::shared_ptr<pooling_forward::desc> pooling_forward_desc;
  pooling_forward_desc.reset(
      new pooling_forward::desc(
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
  //   reorder(input_usr_memory, input_memory).
  //       execute(strm, input_usr_memory, input_memory);
  // }

  // auto expected_output_md = pooling_forward_pd->dst_desc();
  auto output_memory = output_usr_memory;

  // output has the same format with output_usr.
  // if (output_md != expected_output_md) {
  //   output_memory = memory(expected_output_md, engine);
  // }

  auto indices_md = pooling_forward_pd->workspace_desc();
  auto indices_usr_memory =
      memory({{{output_tz}, data_t, format_nchw}, engine});

  auto indices_usr =
      at::empty({output_tz}, at::TensorOptions(kDPCPP).dtype(kInt));
  dpcpp_set_mkldnn_buffer(
      (void*)indices_usr.data_ptr<int32_t>(), indices_usr_memory);
  memory indices_memory = indices_usr_memory;

  std::shared_ptr<pooling_forward> pool_forward;
  pool_forward.reset(new pooling_forward(*pooling_forward_pd));

  // indices has the same format with indices_usr.
  // if (indices_usr_memory.get_desc() != indices_md) {
  //   indices_memory = memory(indices_md, engine);
  // }

  pool_forward->execute(
      strm,
      {{MKLDNN_ARG_SRC, input_memory},
       {MKLDNN_ARG_DST, output_memory},
       {MKLDNN_ARG_WORKSPACE, indices_memory}});

  // reorder output
  // if (output_memory != output_usr_memory) {
  //   reorder(output_memory, output_usr_memory).
  //       execute(strm, output_memory, output_usr_memory);
  // }

  // reorder workgroup

  // if (indices_usr_memory.get_desc() != indices_md) {
  //   reorder(indices_memory, indices_usr_memory).
  //       execute(strm, indices_memory, indices_usr_memory);
  // }

  // reorder(indices_memory, indices_usr_memory).
  //         execute(strm, indices_memory, indices_usr_memory);

  dpcppMemoryCopyType(
      (int64_t*)indices_data,
      indices_usr.data_ptr<int32_t>(),
      indices_usr.numel());
}

void adaptive_max_pool2d_out_template(
    Tensor& output,
    Tensor& indices,
    const Tensor& input,
    IntArrayRef output_size) {
  for (int64_t i = 0; i < input.ndimension(); i++) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_max_pool2d_dpcpp(): expected input to have non-empty spatial "
        "dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      (input.ndimension() == 4),
      "non-empty 4D (batch mode) tensor expected for input");

  TORCH_CHECK(
      output_size.size() == 2,
      "adaptive_max_pool2d: internal error: output_size.size() must be 2");

  int64_t outputHeight = output_size[0];
  int64_t outputWidth = output_size[1];

  Tensor input_ = input.contiguous();
  int64_t nbatch = input_.size(0);
  int64_t nInputPlane = input_.size(1);
  int64_t inputHeight = input_.size(2);
  int64_t inputWidth = input_.size(3);

  TORCH_CHECK(
      (inputHeight % outputHeight == 0),
      "row input size is not divisible by the output size is not supported "
      "yet");
  TORCH_CHECK(
      (inputWidth % outputWidth == 0),
      "column input size is not divisible by the output size is not supported "
      "yet");

  int kH = inputHeight / outputHeight;
  int kW = inputWidth / outputWidth;
  int dH = kH;
  int dW = kW;
  int padW = 0;
  int padH = 0;

  auto alg_kind = algorithm::pooling_max;
  auto prop_kind = dnnl::prop_kind::forward_training;

  output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});
  indices.resize_({nbatch, nInputPlane, outputHeight, outputWidth});

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input_.scalar_type(), "adaptive_max_pool2d", [&] {
        scalar_t* input_data = input_.data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();
        int64_t* indices_data = indices.data_ptr<int64_t>();

        adaptive_max_pool2d_out_frame(
            input_data,
            output_data,
            indices_data,
            nbatch,
            nInputPlane,
            inputWidth,
            inputHeight,
            outputWidth,
            outputHeight,
            kW,
            kH,
            dW,
            dH,
            padW,
            padH,
            alg_kind,
            prop_kind);
      });
}

template <typename scalar_t>
static void adaptive_max_pool2d_backward_out_frame(
    scalar_t* gradInput_data,
    scalar_t* gradOutput_data,
    int64_t* indices_data,
    int64_t nbatch,
    int64_t nPlane,
    int64_t gradInputWidth,
    int64_t gradInputHeight,
    int64_t gradOutputWidth,
    int64_t gradOutputHeight,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    algorithm alg_kind,
    prop_kind prop_kind) {
  at::Device curDevice = at::Device(at::kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  auto data_t = memory::data_type::f32;
  auto format_nchw = memory::format_tag::nchw;

  memory::dims gradInput_tz = {nbatch, nPlane, gradInputHeight, gradInputWidth};
  memory::dims gradOutput_tz = {
      nbatch, nPlane, gradOutputHeight, gradOutputWidth};
  memory::dims kernel = {kH, kW};
  memory::dims stride = {dH, dW};
  memory::dims padding = {padH, padW};

  // Currently, MKLDNN GPU doens't support format_any in pooling
  auto gradInput_md = memory::desc({gradInput_tz}, data_t, format_nchw);
  auto gradOutput_md = memory::desc({gradOutput_tz}, data_t, format_nchw);

  auto diff_dst_usr_memory =
      memory({{{gradOutput_tz}, data_t, format_nchw}, engine});
  dpcpp_set_mkldnn_buffer(gradOutput_data, diff_dst_usr_memory);

  auto diff_src_usr_memory =
      memory({{{gradInput_tz}, data_t, format_nchw}, engine});
  dpcpp_set_mkldnn_buffer(gradInput_data, diff_src_usr_memory);

  std::shared_ptr<pooling_forward::desc> pooling_forward_desc;
  pooling_forward_desc.reset(
      new pooling_forward::desc(
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
  pooling_backward_desc.reset(
      new pooling_backward::desc(
          alg_kind,
          gradInput_md,
          gradOutput_md,
          stride,
          kernel,
          padding,
          padding));
  std::shared_ptr<pooling_backward::primitive_desc> pooling_backward_pd;
  pooling_backward_pd.reset(
      new pooling_backward::primitive_desc(
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
  auto indices_usr_memory = memory(
      {{{gradOutput_tz},
        (memory::data_type)indices_md.data.data_type,
        format_nchw},
       engine});
  dpcpp_set_mkldnn_buffer(
      (void*)indices_usr.data_ptr<int32_t>(), indices_usr_memory);
  auto indices_memory = indices_usr_memory;

  // indices has the same format with indices.
  // reorder indices if needed
  // if (indices_usr_memory.get_desc() != indices_md) {
  //   reorder(indices_usr_memory, indices_memory).
  //       execute(strm, indices_usr_memory, indices_memory);
  // }
  pool_backward->execute(
      strm,
      {{MKLDNN_ARG_DIFF_DST, diff_dst_memory},
       {MKLDNN_ARG_DIFF_SRC, diff_src_memory},
       {MKLDNN_ARG_WORKSPACE, indices_memory}});

  // Reorder diff_src
  // if (diff_src_memory != diff_src_usr_memory) {
  //   reorder(diff_src_memory, diff_src_usr_memory).
  //       execute(strm, diff_src_memory, diff_src_usr_memory);
  // }
}

Tensor& adaptive_max_pool2d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input,
    const Tensor& indices) {
  TORCH_CHECK(
      input.ndimension() == 4, "only support 4 dims on DPCPP device now!");
  Tensor gradOutput = gradOutput_.contiguous();

  int64_t nbatch = input.size(0);
  int64_t nPlane = input.size(1);
  int64_t gradInputHeight = input.size(2);
  int64_t gradInputWidth = input.size(3);

  int64_t gradOutputHeight = gradOutput.size(2);
  int64_t gradOutputWidth = gradOutput.size(3);

  TORCH_CHECK(
      (gradInputHeight % gradOutputHeight == 0),
      "row input size is not divisible by the output size is not supported "
      "yet");
  TORCH_CHECK(
      (gradInputWidth % gradOutputWidth == 0),
      "column input size is not divisible by the output size is not supported "
      "yet");

  int padW = 0;
  int padH = 0;
  int kH = gradInputHeight / gradOutputHeight;
  int kW = gradInputWidth / gradOutputWidth;
  int dH = kH;
  int dW = kW;

  gradInput.resize_as_(input);
  gradInput.zero_();

  auto alg_kind = algorithm::pooling_max;
  auto prop_kind = dnnl::prop_kind::forward_training;

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "adaptive_max_pool2d_backward", [&] {
        /* get raw pointers */
        scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
        scalar_t* gradOutput_data = gradOutput.data_ptr<scalar_t>();
        int64_t* indices_data = indices.data_ptr<int64_t>();

        adaptive_max_pool2d_backward_out_frame<scalar_t>(
            gradInput_data,
            gradOutput_data,
            indices_data,
            nbatch,
            nPlane,
            gradInputWidth,
            gradInputHeight,
            gradOutputWidth,
            gradOutputHeight,
            kW,
            kH,
            dW,
            dH,
            padH,
            padW,
            alg_kind,
            prop_kind);
      });
  return gradInput;
}

} // impl

std::tuple<Tensor&, Tensor&> adaptive_max_pool2d_out(
    Tensor& out,
    Tensor& indices,
    const Tensor& self,
    IntArrayRef output_size) {
  impl::adaptive_max_pool2d_out_template(out, indices, self, output_size);
  return std::tuple<Tensor&, Tensor&>(out, indices);
}

std::tuple<Tensor, Tensor> adaptive_max_pool2d(
    const Tensor& self,
    IntArrayRef output_size) {
  Tensor output = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  TORCH_INTERNAL_ASSERT(output_size.size() == 2);
  return at::AtenIpexTypeDPCPP::adaptive_max_pool2d_out(
      output, indices, self, output_size);
}

Tensor& adaptive_max_pool2d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices) {
  impl::adaptive_max_pool2d_backward_out_template(
      grad_input, grad_output, self, indices);
  return grad_input;
}

Tensor adaptive_max_pool2d_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices) {
  auto grad_input = at::zeros_like(self);
  return at::AtenIpexTypeDPCPP::adaptive_max_pool2d_backward_out(
      grad_input, grad_output, self, indices);
}

} // AtenIpexTypeDPCPP
} // at
