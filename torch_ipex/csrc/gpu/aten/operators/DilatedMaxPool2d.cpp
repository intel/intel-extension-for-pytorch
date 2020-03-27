#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <tuple>

#include <core/Runtime.h>
#include <utils/Math.h>

using namespace mkldnn;
using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

template <typename scalar_t>
static void max_pool2d_with_indices_out_frame(
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
  at::Device curDevice = at::Device(at::kDPCPP, current_device());
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

  // Currently, DPCPP path doesn't support internal format.
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

template <typename scalar_t>
static void max_pool2d_with_indices_backward_out_frame(
    scalar_t* gradInput_data,
    scalar_t* gradOutput_data,
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
  at::Device curDevice = at::Device(at::kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  auto data_t = memory::data_type::f32;
  auto format_nchw = memory::format_tag::nchw;

  memory::dims input_tz = {nbatch, nInputPlane, inputHeight, inputWidth};
  memory::dims output_tz = {nbatch, nInputPlane, outputHeight, outputWidth};
  memory::dims kernel = {kH, kW};
  memory::dims stride = {dH, dW};
  memory::dims padding = {padH, padW};

  // Currently, MKLDNN GPU doens't support format_any in pooling
  auto input_md = memory::desc({input_tz}, data_t, format_nchw);
  auto output_md = memory::desc({output_tz}, data_t, format_nchw);

  auto diff_dst_usr_memory =
      memory({{{output_tz}, data_t, format_nchw}, engine});
  dpcpp_set_mkldnn_buffer(gradOutput_data, diff_dst_usr_memory);

  auto diff_src_usr_memory =
      memory({{{input_tz}, data_t, format_nchw}, engine});
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

  // Currently, DPCPP path doesn't support internal format.
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
      at::empty({output_tz}, at::TensorOptions(kDPCPP).dtype(kInt));
  dpcppMemoryCopyType(
      indices_usr.data_ptr<int32_t>(),
      (int64_t*)indices_data,
      indices_usr.numel());

  pool_backward.reset(new pooling_backward(*pooling_backward_pd));

  auto indices_md = pooling_forward_pd->workspace_desc();
  auto indices_usr_memory = memory(
      {{{output_tz}, (memory::data_type)indices_md.data.data_type, format_nchw},
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

void max_pool2d_with_indices_out_template(
    Tensor& output,
    Tensor& indices,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty()
      ? kW
      : stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple "
      "of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of "
      "two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK(
      input_.ndimension() == 4, "only support 4 dims on DPCPP device now!");

  /* sizes */
  const int64_t nbatch = input_.size(-4);
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  const int64_t outputHeight = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);

  pool2d_shape_check(
      input_,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth);

  /* get contiguous input */
  Tensor input = input_.contiguous();
  output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});
  /* indices will contain the locations for each output point */
  indices.resize_({nbatch, nInputPlane, outputHeight, outputWidth});

  auto alg_kind = algorithm::pooling_max;
  auto prop_kind = dnnl::prop_kind::forward_training;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "max_pool2d_with_indices", [&] {
        scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();
        int64_t* indices_data = indices.data_ptr<int64_t>();

        max_pool2d_with_indices_out_frame(
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

Tensor& max_pool2d_with_indices_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty()
      ? kW
      : stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple "
      "of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of "
      "two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK(
      input.ndimension() == 4, "only support 4 dims on DPCPP device now!");

  /* get contiguous gradOutput */
  const Tensor gradOutput = gradOutput_.contiguous();

  /* resize */
  gradInput.resize_as_(input);
  gradInput.zero_();

  /* sizes */
  const int64_t nbatch = input.size(-4);
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);
  const int64_t outputHeight = gradOutput.size(-2);
  const int64_t outputWidth = gradOutput.size(-1);

  /* XXX preserve the existing shape check behavior */
  const int64_t outputHeight_for_shape_check = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth_for_shape_check = pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);

  auto alg_kind = algorithm::pooling_max;
  auto prop_kind = dnnl::prop_kind::forward_training;

  max_pool2d_backward_shape_check(
      input,
      gradOutput_,
      indices,
      nbatch,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight_for_shape_check,
      outputWidth_for_shape_check);

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "max_pool2d_with_indices_backward", [&] {
        /* get raw pointers */
        scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
        scalar_t* gradOutput_data = gradOutput.data_ptr<scalar_t>();
        int64_t* indices_data = indices.data_ptr<int64_t>();

        max_pool2d_with_indices_backward_out_frame<scalar_t>(
            gradInput_data,
            gradOutput_data,
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
            padH,
            padW,
            alg_kind,
            prop_kind);
      });
  return gradInput;
}

} // namespace impl

std::tuple<Tensor&, Tensor&> max_pool2d_with_indices_out(
    Tensor& output,
    Tensor& indices,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  impl::max_pool2d_with_indices_out_template(
      output,
      indices,
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> max_pool2d_with_indices(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  Tensor output = at::empty({0}, input.options());
  Tensor indices = at::empty({0}, input.options().dtype(kLong));
  return at::AtenIpexTypeDPCPP::max_pool2d_with_indices_out(
      output,
      indices,
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
}

Tensor& max_pool2d_with_indices_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  impl::max_pool2d_with_indices_backward_out_template(
      grad_input,
      grad_output,
      self,
      indices,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
  return grad_input;
}

Tensor max_pool2d_with_indices_backward(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  auto grad_input = at::zeros_like(self);
  return at::AtenIpexTypeDPCPP::max_pool2d_with_indices_backward_out(
      grad_input,
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      indices);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
