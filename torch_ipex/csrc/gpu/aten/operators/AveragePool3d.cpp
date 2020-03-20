#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <core/Runtime.h>
#include <vector>

using namespace dnnl;
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
    AT_ERROR(" mkldnn not support for double");
  };
};

template <typename scalar_t>
static void avg_pool3d_out_frame(
    scalar_t* input_data,
    scalar_t* output_data,
    int64_t nbatch,
    int64_t nblock,
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
  auto format = memory::format_tag::ncdhw;

  memory::dims input_tz = {nbatch, nblock, inputDepth, inputHeight, inputWidth};
  memory::dims output_tz = {
      nbatch, nblock, outputDepth, outputHeight, outputWidth};
  memory::dims kernel = {kD, kH, kW};
  memory::dims stride = {dD, dH, dW};
  memory::dims padding = {padD, padH, padW};

  auto input_md = memory::desc({input_tz}, data_t, format);
  auto output_md = memory::desc({output_tz}, data_t, format);

  auto input_usr_memory = memory({{{input_tz}, data_t, format}, engine});
  dpcpp_set_mkldnn_buffer(input_data, input_usr_memory);

  auto output_usr_memory = memory({{{output_tz}, data_t, format}, engine});
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
static void avg_pool3d_backward_out_frame(
    scalar_t* gradInput_data,
    scalar_t* gradOutput_data,
    int64_t nbatch,
    int64_t nblock,
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
  auto format = memory::format_tag::ncdhw;

  memory::dims input_tz = {nbatch, nblock, inputDepth, inputHeight, inputWidth};
  memory::dims output_tz = {
      nbatch, nblock, outputDepth, outputHeight, outputWidth};
  memory::dims kernel = {kD, kH, kW};
  memory::dims stride = {dD, dH, dW};
  memory::dims padding = {padD, padH, padW};

  auto input_md = memory::desc({input_tz}, data_t, format);
  auto output_md = memory::desc({output_tz}, data_t, format);

  auto diff_dst_usr_memory = memory({{{output_tz}, data_t, format}, engine});
  dpcpp_set_mkldnn_buffer(gradOutput_data, diff_dst_usr_memory);

  auto diff_src_usr_memory = memory({{{input_tz}, data_t, format}, engine});
  dpcpp_set_mkldnn_buffer(gradInput_data, diff_src_usr_memory);

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

  std::shared_ptr<pooling_backward::desc> pooling_backward_desc;
  pooling_backward_desc.reset(
      new pooling_backward::desc(
          alg_kind, input_md, output_md, stride, kernel, padding, padding));
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

void avg_pool3d_out_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) {
  //  TensorArg output_arg{ output, "output", 1 };
  //  TensorArg input_arg{ input, "input", 2 };
  //
  //  checkAllSameGPU("avg_pool3d_out_sycl", {output_arg, input_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must either be a single int, or a tuple of "
      "three ints");
  const int kD = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d: stride must either be omitted, a single int, or a tuple of "
      "three ints");
  const int dD = stride.empty() ? kD : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH : stride.size() == 1
          ? dD
          : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW : stride.size() == 1
          ? dD
          : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must either be a single int, or a tuple of three "
      "ints");
  const int padD = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nblock = input.size(-4);
  const int64_t idepth = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  const int64_t outputDepth =
      pooling_output_shape<int64_t>(idepth, kD, padD, dD, 1, ceil_mode);
  const int64_t outputHeight =
      pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth =
      pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  pool3d_shape_check(
      input,
      nblock,
      kD,
      kH,
      kW,
      dD,
      dH,
      dW,
      padD,
      padH,
      padW,
      1,
      1,
      1,
      idepth,
      iheight,
      iwidth,
      outputDepth,
      outputHeight,
      outputWidth,
      /*check_input_size=*/true);

  if (input.ndimension() == 4) {
    output.resize_({nblock, outputDepth, outputHeight, outputWidth});
  } else {
    output.resize_({nbatch, nblock, outputDepth, outputHeight, outputWidth});
  }

  TORCH_CHECK(output.is_contiguous(), "avg_pool3d: output must be contiguous");

  Tensor work_input = input.contiguous();

  auto alg_kind = count_include_pad ? algorithm::pooling_avg_include_padding
                                    : algorithm::pooling_avg_exclude_padding;
  auto prop_kind = dnnl::prop_kind::forward_training;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "avg_pool3d_frame", [&] {
        scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();

        avg_pool3d_out_frame(
            input_data,
            output_data,
            nbatch,
            nblock,
            idepth,
            iheight,
            iwidth,
            outputDepth,
            outputHeight,
            outputWidth,
            kD,
            kH,
            kW,
            dD,
            dH,
            dW,
            padD,
            padH,
            padW,
            alg_kind,
            prop_kind);
      });
}

Tensor& avg_pool3d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) {
  //  TensorArg gradInput_arg{ gradInput, "gradInput", 1 };
  //  TensorArg gradOutput_arg{ gradOutput, "gradOutput", 2 };
  //  TensorArg input_arg{ input, "input", 3 };
  //
  //  checkAllSameGPU("avg_pool3d_backward_out_sycl",
  //                  {gradInput_arg, gradOutput_arg, input_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must either be a single int, or a tuple of "
      "three ints");
  const int kD = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d: stride must either be omitted, a single int, or a tuple of "
      "three ints");
  const int dD = stride.empty() ? kD : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH : stride.size() == 1
          ? dD
          : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW : stride.size() == 1
          ? dD
          : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must either be a single int, or a tuple of three "
      "ints");
  const int padD = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(
      (gradOutput.ndimension() == 4 || gradOutput.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for gradOutput");

  /* resize */
  gradInput.resize_as_(input);
  gradInput.zero_();
  TORCH_CHECK(gradInput.is_contiguous(), "gradInput must be contiguous");

  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nblock = input.size(-4);
  const int64_t idepth = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  const int64_t odepth = gradOutput.size(-3);
  const int64_t oheight = gradOutput.size(-2);
  const int64_t owidth = gradOutput.size(-1);

  /* XXX shape check behavior from TH */
  const int64_t odepth_for_shape_check =
      pooling_output_shape<int64_t>(idepth, kD, padD, dD, 1, ceil_mode);
  const int64_t oheight_for_shape_check =
      pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth_for_chape_check =
      pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  avg_pool3d_backward_shape_check(
      input,
      gradOutput,
      nblock,
      kD,
      kH,
      kW,
      dD,
      dH,
      dW,
      padD,
      padH,
      padW,
      idepth,
      iheight,
      iwidth,
      odepth,
      oheight,
      owidth);

  Tensor work_grad_input = gradInput;
  Tensor work_grad_output = gradOutput.contiguous();

  if (input.ndimension() == 5) {
    // Collapse batch and feature dimensions.
    work_grad_input =
        work_grad_input.reshape({nbatch * nblock, idepth, iheight, iwidth});
    work_grad_output =
        work_grad_output.reshape({nbatch * nblock, odepth, oheight, owidth});
  }

  auto alg_kind = count_include_pad ? algorithm::pooling_avg_include_padding
                                    : algorithm::pooling_avg_exclude_padding;
  auto prop_kind = dnnl::prop_kind::forward_training;

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "avg_pool3d_backward_out_frame", [&] {
        scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
        scalar_t* gradOutput_data = gradOutput.data_ptr<scalar_t>();

        avg_pool3d_backward_out_frame(
            gradInput_data,
            gradOutput_data,
            nbatch,
            nblock,
            idepth,
            iheight,
            iwidth,
            odepth,
            oheight,
            owidth,
            kD,
            kH,
            kW,
            dD,
            dH,
            dW,
            padD,
            padH,
            padW,
            alg_kind,
            prop_kind);
      });

  return gradInput;
}
} // namespace impl

Tensor& avg_pool3d_out(
    Tensor& out,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      !divisor_override.has_value(),
      "dpcpp_avg_pool3d operator does not support divisor");
  impl::avg_pool3d_out_template(
      out, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  return out;
}

Tensor avg_pool3d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  Tensor output = at::empty({0}, self.options());
  return at::AtenIpexTypeDPCPP::avg_pool3d_out(
      output,
      self,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
}

Tensor& avg_pool3d_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      !divisor_override.has_value(),
      "dpcpp_avg_pool3d operator does not support divisor");
  impl::avg_pool3d_backward_out_template(
      grad_input,
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad);
  return grad_input;
}

Tensor avg_pool3d_backward(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  auto grad_input = at::zeros_like(self);
  return at::AtenIpexTypeDPCPP::avg_pool3d_backward_out(
      grad_input,
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
