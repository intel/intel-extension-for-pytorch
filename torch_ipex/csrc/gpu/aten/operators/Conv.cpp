#include <ATen/quantized/QTensorImpl.h>
#include <ATen/ipex_type_dpcpp_customized.h>
#include <ATen/quantized/Quantizer.h>
#include <core/DPCPPUtils.h>
#include <core/Quantizer.h>
#include <core/Runtime.h>
#include <core/TensorImplUtils.h>
#include <tensor/Context.h>

#include <utils/ParamUtils.h>
#include <oneDNN/oneDNN.h>

#include "ConvTranspose.h"

using namespace dnnl;
using namespace at::dpcpp;
using namespace at::native;
using namespace at::xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

Tensor dpcpp_convolution_backward_input(
    IntArrayRef input_size,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined) {
  auto engine = GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto grad_input = at::empty(input_size, grad_output.options());
  if (grad_input.numel() == 0) {
    return grad_input;
  }

  auto ndim = grad_input.ndimension();
  auto ic = grad_input.size(1);
  auto oc = grad_output.size(1);

  // align data type with bf16
  auto data_grad = get_onednn_dtype(grad_output);
  auto weight_t = get_onednn_dtype(weight);
  auto bias_t = dnnl::memory::data_type::f32;
  auto weight_usr_t = weight_t;
  auto format_any = memory::format_tag::any;
  auto format_input = conv_src_fmt(ndim);
  auto format_weight = conv_wgh_fmt(ndim, groups != 1);

  memory::dims input_tz = grad_input.sizes().vec();
  memory::dims weight_tz = compatible_wgh_dims(ndim, groups, oc, ic, weight.sizes());
  memory::dims bias_tz = {oc};
  memory::dims output_tz = grad_output.sizes().vec();
  output_tz[0] = grad_input.size(0); // set n

  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = compatible_dilation(dilation);

  //Master weight
  if (data_grad == dnnl::memory::data_type::bf16) {
      weight_t = dnnl::memory::data_type::bf16;
      bias_t = dnnl::memory::data_type::bf16;
  }

  auto input_md = memory::desc(input_tz, data_grad, format_any);
  auto weight_md = memory::desc(weight_tz, weight_t, format_any);
  auto output_md = memory::desc(output_tz, data_grad, format_any);
  auto bias_md = bias_defined ? memory::desc(bias_tz, bias_t, format_any) : memory::desc();

  auto conv_forward_desc = convolution_forward::desc(
      prop_kind::forward, algorithm::convolution_direct,
      input_md, weight_md, bias_md, output_md,
      _stride, _dilation, _padding, _padding);

  auto conv_forward_pd = convolution_forward::primitive_desc(conv_forward_desc, engine);

  auto conv_backward_data_desc = convolution_backward_data::desc(
      algorithm::convolution_direct,
      input_md, weight_md, output_md,
      _stride, _dilation, _padding, _padding);

  auto conv_backward_data_pd = convolution_backward_data::primitive_desc(
      conv_backward_data_desc, engine, conv_forward_pd);

  memory grad_output_usr_memory, weight_usr_memory, grad_input_usr_memory;
  if (!lazy_reorder_enabled()) {
    grad_output_usr_memory = dpcpp_onednn_memory(
      {output_tz, data_grad, format_input}, engine, grad_output.data_ptr());

    weight_usr_memory = dpcpp_onednn_memory(
      {weight_tz, weight_usr_t, format_weight}, engine, weight.data_ptr());

    grad_input_usr_memory = dpcpp_onednn_memory(
      {input_tz, data_grad, format_input}, engine, grad_input.data_ptr());
  } else {
    auto grad_output_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(grad_output);
    grad_output_usr_memory = grad_output_ctx.is_plain()
        ? dpcpp_onednn_memory({output_tz, data_grad, format_input}, engine, grad_output.data_ptr())
        : dpcpp_onednn_memory({grad_output_ctx.meta()}, engine, grad_output.data_ptr());

    auto weight_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(weight);
    weight_usr_memory = weight_ctx.is_plain()
        ? dpcpp_onednn_memory({weight_tz, weight_usr_t, format_weight}, engine, weight.data_ptr())
        : dpcpp_onednn_memory({weight_ctx.meta()}, engine, weight.data_ptr());

    auto grad_input_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(grad_input);
    grad_input_usr_memory = grad_input_ctx.is_plain()
        ? dpcpp_onednn_memory({output_tz, data_grad, format_input}, engine, grad_input.data_ptr())
        : dpcpp_onednn_memory({grad_input_ctx.meta()}, engine, grad_input.data_ptr());
  }

  Tensor grad_output_;
  auto expected_grad_output_md = conv_backward_data_pd.diff_dst_desc();
  auto grad_output_memory = grad_output_usr_memory;
  if (grad_output_usr_memory.get_desc() != expected_grad_output_md) {
    auto item_num = static_cast<int64_t>(expected_grad_output_md.get_size() / grad_output.itemsize());
    grad_output_ = at::AtenIpexTypeXPU::empty({item_num}, grad_output.options(), c10::nullopt);
    grad_output_memory = dpcpp_onednn_memory(expected_grad_output_md, engine, grad_output_.data_ptr());
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(grad_output_usr_memory, grad_output_memory),
        strm,
        {{DNNL_ARG_FROM, grad_output_usr_memory},
        {DNNL_ARG_TO, grad_output_memory}});
  }

  Tensor weight_;
  auto expected_weight_md = conv_backward_data_pd.weights_desc();
  auto weight_memory = weight_usr_memory;
  if (weight_usr_memory.get_desc() != expected_weight_md) {
    auto item_num = static_cast<int64_t>(expected_weight_md.get_size() / weight.itemsize());
    weight_ = at::AtenIpexTypeXPU::empty({item_num}, weight.options(), c10::nullopt);
    weight_memory = dpcpp_onednn_memory(expected_weight_md, engine, weight_.data_ptr());
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(weight_usr_memory, weight_memory),
        strm,
        {{DNNL_ARG_FROM, weight_usr_memory},
        {DNNL_ARG_TO, weight_memory}});
  }

  Tensor grad_input_;
  auto expected_grad_input_md = conv_backward_data_pd.diff_src_desc();
  auto grad_input_memory = grad_input_usr_memory;
  if (grad_input_memory.get_desc() != expected_grad_input_md) {
    grad_input_ = empty_opaque_tensor(expected_grad_input_md, grad_input.options(), c10::nullopt);
    grad_input_memory = dpcpp_onednn_memory(expected_grad_input_md, engine, grad_input_.data_ptr());
  }

  auto conv_backward_data = convolution_backward_data(conv_backward_data_pd);
  DPCPP_ONEDNN_EXEC(
      conv_backward_data,
      strm,
      {{DNNL_ARG_DIFF_DST, grad_output_memory},
      {DNNL_ARG_WEIGHTS, weight_memory},
      {DNNL_ARG_DIFF_SRC, grad_input_memory}});

  if (!lazy_reorder_enabled() && grad_input_memory != grad_input_usr_memory) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(grad_input_memory, grad_input_usr_memory),
        strm,
        {{DNNL_ARG_FROM, grad_input_memory},
        {DNNL_ARG_TO, grad_input_usr_memory}});
  } else if (lazy_reorder_enabled() && grad_input_memory != grad_input_usr_memory) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(grad_input_);
    DPCPPTensorContext::set_tensor_ctx(grad_input, std::move(blk_ctx));
  }

  return grad_input;
}

std::tuple<at::Tensor, at::Tensor> dpcpp_convolution_backward_weights(
    IntArrayRef weight_size,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined) {
  auto engine = GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  Tensor grad_bias;
  if (bias_defined) {
    grad_bias = at::empty({grad_output.size(1)}, grad_output.options());
  }

  auto grad_weight = at::empty(weight_size, grad_output.options());
  if (input.numel() == 0) {
    return std::tuple<at::Tensor, at::Tensor>{grad_weight, grad_bias};
  }

  auto ndim = input.ndimension();
  auto ic = input.size(1);
  auto oc = grad_output.size(1);

  // align data type with bf16
  auto data_grad = get_onednn_dtype(grad_output);
  auto weight_t = data_grad;
  auto bias_t = memory::data_type::f32;
  auto format_any = memory::format_tag::any;
  auto format_input = conv_src_fmt(ndim);
  auto format_weight = conv_wgh_fmt(ndim, groups != 1);
  auto format_bias = memory::format_tag::x;

  // Naive Master weight - We keep BF16 gw and gb output, and reorder them to fp32 in SGD.
  auto grad_weight_t = weight_t;
  auto grad_bias_t = weight_t;

  memory::dims input_tz = input.sizes().vec();
  memory::dims weight_tz = compatible_wgh_dims(ndim, groups, oc, ic, grad_weight.sizes());
  memory::dims bias_tz = {oc};
  memory::dims output_tz = grad_output.sizes().vec();
  output_tz[0] = input.size(0); // set n

  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = compatible_dilation(dilation);

  auto input_md = memory::desc(input_tz, data_grad, format_any);
  // Master weight - for now, we want plain gw output and gb output because weight and bias in sgd is plain.
  //                 while plain calculation in Conv3d cannot get the correct gw in PreCI.
  //                 Thus, we still use format_any here and add one reorder in the end.
  auto weight_md = memory::desc(weight_tz, grad_weight_t, format_any);
  auto bias_md = bias_defined ? memory::desc(bias_tz, grad_bias_t, format_any) : memory::desc();

  auto output_md = memory::desc(output_tz, weight_t, format_any);
  auto conv_forward_desc = convolution_forward::desc(
      prop_kind::forward, algorithm::convolution_direct,
      input_md, weight_md, bias_md, output_md,
      _stride, _dilation, _padding, _padding);

  auto conv_forward_pd = convolution_forward::primitive_desc(conv_forward_desc, engine);

  auto conv_backward_weight_desc = convolution_backward_weights::desc(
          algorithm::convolution_direct,
          input_md, weight_md, bias_md, output_md,
          _stride, _dilation, _padding, _padding);

  auto conv_backward_weight_pd = convolution_backward_weights::primitive_desc(
          conv_backward_weight_desc, engine, conv_forward_pd);

  memory input_usr_memory, grad_output_usr_memory, grad_weight_usr_memory;
  if (!lazy_reorder_enabled()) {
    input_usr_memory = dpcpp_onednn_memory(
        {input_tz, data_grad, format_input}, engine, input.data_ptr());

    grad_output_usr_memory = dpcpp_onednn_memory(
        {output_tz, data_grad, format_input}, engine, grad_output.data_ptr());

    grad_weight_usr_memory = dpcpp_onednn_memory(
        {weight_tz, grad_weight_t, format_weight}, engine, grad_weight.data_ptr());
  } else {
    auto input_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(input);
    input_usr_memory = input_ctx.is_plain()
        ? dpcpp_onednn_memory({input_tz, data_grad, format_input}, engine, input.data_ptr())
        : dpcpp_onednn_memory({input_ctx.meta()}, engine, input.data_ptr());

    auto grad_output_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(grad_output);
    grad_output_usr_memory = grad_output_ctx.is_plain()
        ? dpcpp_onednn_memory({output_tz, data_grad, format_input}, engine, grad_output.data_ptr())
        : dpcpp_onednn_memory({grad_output_ctx.meta()}, engine, grad_output.data_ptr());

    auto grad_weight_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(grad_weight);
    grad_weight_usr_memory = grad_weight_ctx.is_plain()
        ? dpcpp_onednn_memory({weight_tz, grad_weight_t, format_weight}, engine, grad_weight.data_ptr())
        : dpcpp_onednn_memory({grad_weight_ctx.meta()}, engine, grad_weight.data_ptr());
  }

  Tensor input_;
  auto expected_input_md = conv_backward_weight_pd.src_desc();
  auto input_memory = input_usr_memory;
  if (input_usr_memory.get_desc() != expected_input_md) {
    auto item_num = static_cast<int64_t>(expected_input_md.get_size() / input.itemsize());
    input_ = at::AtenIpexTypeXPU::empty({item_num}, input.options(), c10::nullopt);
    input_memory = dpcpp_onednn_memory(expected_input_md, engine, input_.data_ptr());
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(input_usr_memory, input_memory),
        strm,
        {{DNNL_ARG_FROM, input_usr_memory},
        {DNNL_ARG_TO, input_memory}});
  }

  Tensor grad_output_;
  auto expected_grad_output_md = conv_backward_weight_pd.diff_dst_desc();
  auto grad_output_memory = grad_output_usr_memory;
  if (grad_output_usr_memory.get_desc() != expected_grad_output_md) {
    auto item_num = static_cast<int64_t>(expected_grad_output_md.get_size() / grad_output.itemsize());
    grad_output_ = at::AtenIpexTypeXPU::empty({item_num}, grad_output.options(), c10::nullopt);
    grad_output_memory = dpcpp_onednn_memory(expected_grad_output_md, engine, grad_output_.data_ptr());
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(grad_output_usr_memory, grad_output_memory),
        strm,
        {{DNNL_ARG_FROM, grad_output_usr_memory},
        {DNNL_ARG_TO, grad_output_memory}});
  }

  Tensor grad_weight_;
  auto expected_grad_weight_md = conv_backward_weight_pd.diff_weights_desc();
  auto grad_weight_memory = grad_weight_usr_memory;
  if (grad_weight_usr_memory.get_desc() != expected_grad_weight_md) {
    auto item_num = static_cast<int64_t>(expected_grad_weight_md.get_size() / grad_weight.itemsize());
    grad_weight_ = at::AtenIpexTypeXPU::empty({item_num}, grad_weight.options(), c10::nullopt);
    grad_weight_memory = dpcpp_onednn_memory(expected_grad_weight_md, engine, grad_weight_.data_ptr());
  }

  memory grad_bias_memory = memory({{}, bias_t, format_bias}, engine);
  if (bias_defined) {
    if (!lazy_reorder_enabled()) {
      grad_bias_memory = dpcpp_onednn_memory(
        {bias_tz, bias_t, format_bias}, engine, grad_bias.data_ptr());
    } else {
      auto grad_bias_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(grad_bias);
      grad_bias_memory = grad_bias_ctx.is_plain()
        ? dpcpp_onednn_memory({bias_tz, grad_bias_t, format_bias}, engine, grad_bias.data_ptr())
        : dpcpp_onednn_memory({grad_bias_ctx.meta()}, engine, grad_bias.data_ptr());
    }
  }

  auto conv_backward_weight = convolution_backward_weights(conv_backward_weight_pd);
  DPCPP_ONEDNN_EXEC(
      conv_backward_weight,
      strm,
      {{DNNL_ARG_DIFF_DST, grad_output_memory},
      {DNNL_ARG_SRC, input_memory},
      {DNNL_ARG_DIFF_WEIGHTS, grad_weight_memory},
      {DNNL_ARG_DIFF_BIAS, grad_bias_memory}});

  if (grad_weight_memory.get_desc() != grad_weight_usr_memory.get_desc()) {
    // grad_weight_ contains the result of gw backward, while it is blk format.
    // In training mode, plain gw output is expected for sgd update regardless of lazy_reorder_enabled or not.
    // Thus, we need one additional reorder here to make grad_weight plain.
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(grad_weight_memory, grad_weight_usr_memory),
        strm,
        {{DNNL_ARG_FROM, grad_weight_memory},
        {DNNL_ARG_TO, grad_weight_usr_memory}});
  }
  return std::tuple<at::Tensor, at::Tensor>{grad_weight, grad_bias};
}

struct ConvParams {
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool transposed;
  std::vector<int64_t> output_padding;
  int groups;
  bool benchmark;
  bool deterministic;

  bool is_strided() const;
  bool is_dilated() const;
  bool is_padded() const;
  bool is_output_padding_neg() const;
  bool is_output_padding_big() const;
  bool is_padding_neg() const;
  bool is_stride_nonpos() const;
  void view1d_as_2d();
  bool use_cpu_depthwise3x3_winograd(
      const at::Tensor& input,
      const at::Tensor& weight) const;
  bool is_depthwise(const at::Tensor& input, const at::Tensor& weight) const;
};

std::ostream& operator<<(std::ostream& out, const ConvParams& params) {
  out << "ConvParams {"
      << "  stride = " << IntArrayRef{params.stride}
      << "  padding = " << IntArrayRef{params.padding}
      << "  dilation = " << IntArrayRef{params.dilation}
      << "  transposed = " << params.transposed
      << "  output_padding = " << IntArrayRef{params.output_padding}
      << "  groups = " << params.groups << "  benchmark = " << params.benchmark
      << "  deterministic = " << params.deterministic << "}";
  return out;
}

bool ConvParams::is_strided() const {
  bool is_strided = false;
  for (int s : stride) {
    is_strided |= (s != 1);
  }
  return is_strided;
}

bool ConvParams::is_dilated() const {
  bool is_dilated = false;
  for (int d : dilation) {
    is_dilated |= (d != 1);
  }
  return is_dilated;
}

bool ConvParams::is_padded() const {
  bool is_padded = false;
  for (int p : padding) {
    is_padded |= (p != 0);
  }
  return is_padded;
}

bool ConvParams::is_output_padding_neg() const {
  bool is_non_neg = false;
  for (int p : output_padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}

bool ConvParams::is_output_padding_big() const {
  bool is_big = false;
  for (size_t i = 0; i < output_padding.size(); i++) {
    is_big |=
        (output_padding[i] >= stride[i] || output_padding[i] >= dilation[i]);
  }
  return is_big;
}

bool ConvParams::is_padding_neg() const {
  bool is_non_neg = false;
  for (int p : padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}

bool ConvParams::is_stride_nonpos() const {
  bool is_nonpos = false;
  for (int s : stride) {
    is_nonpos |= (s <= 0);
  }
  return is_nonpos;
}

void ConvParams::view1d_as_2d() {
  if (stride.size() == 1) {
    stride.insert(stride.begin(), 1);
    padding.insert(padding.begin(), 0);
    dilation.insert(dilation.begin(), 1);
    output_padding.insert(output_padding.begin(), 0);
  }
}

bool ConvParams::use_cpu_depthwise3x3_winograd(
    const at::Tensor& input,
    const at::Tensor& weight) const {
  return false;
}

bool ConvParams::is_depthwise(const at::Tensor& input, const at::Tensor& weight) const {
  return !transposed && input.ndimension() == 4 && input.size(1) == groups &&
      groups > 1 && // no point if there is only a single group
      weight.size(0) % input.size(1) ==
      0; // output channels must be a multiple of input channels
}

static void check_shape_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const ConvParams& params,
    bool input_is_mkldnn) {
  int64_t k = input.ndimension();
  int64_t weight_dim = weight.ndimension();
  std::vector<int64_t> weight_sizes(weight_dim);
  if ((weight_dim == k + 1) && input_is_mkldnn) {
    weight_sizes[0] = weight.size(0) * weight.size(1);
    std::copy_n(weight.sizes().cbegin() + 2, k - 1, weight_sizes.begin() + 1);
    weight_dim = k;
  } else {
    std::copy_n(weight.sizes().cbegin(), weight_dim, weight_sizes.begin());
  }
  int64_t groups = params.groups;
  auto padding = params.padding;
  auto output_padding = params.output_padding;
  auto stride = params.stride;
  auto dilation = params.dilation;
  bool transposed = params.transposed;

  TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported");
  TORCH_CHECK(
      !params.is_output_padding_neg(),
      "negative output_padding is not supported");
  TORCH_CHECK(
      !params.is_stride_nonpos(), "non-positive stride is not supported");

  TORCH_CHECK(
      weight_dim == k,
      "Expected ",
      weight_dim,
      "-dimensional input for ",
      weight_dim,
      "-dimensional weight ",
      weight_sizes,
      ", but got ",
      k,
      "-dimensional input of size ",
      input.sizes(),
      " instead");
  TORCH_CHECK(
      weight_sizes[0] >= groups,
      "Given groups=",
      groups,
      ", expected weight to be at least ",
      groups,
      " at dimension 0, but got weight of size ",
      weight_sizes,
      " instead");
  TORCH_CHECK(
      weight_sizes[0] % groups == 0,
      "Given groups=",
      groups,
      ", expected weight to be divisible by ",
      groups,
      " at dimension 0, but got weight of size ",
      weight_sizes,
      " instead");

  if (!transposed) {
    std::vector<int64_t> input_shape;
    std::vector<int64_t> kernel_shape;
    bool kernel_size_correct = true;

    TORCH_CHECK(
        input.size(1) == (weight_sizes[1] * groups),
        "Given groups=",
        groups,
        ", weight of size ",
        weight_sizes,
        ", expected input",
        input.sizes(),
        " to have ",
        (weight_sizes[1] * groups),
        " channels, but got ",
        input.size(1),
        " channels instead");
    TORCH_CHECK(
        !bias.defined() ||
            (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
        "Given weight of size ",
        weight_sizes,
        ", expected bias to be 1-dimensional with ",
        weight_sizes[0],
        " elements",
        ", but got bias of size ",
        bias.sizes(),
        " instead");

    for (int i = 2; i < k; ++i) {
      input_shape.push_back(input.size(i) + 2 * padding[i - 2]);
      kernel_shape.push_back(dilation[i - 2] * (weight_sizes[i] - 1) + 1);
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }

    TORCH_CHECK(
        input_shape.size() == kernel_shape.size(),
        "Inconsistent shape between Input and Kernel");

    if (!kernel_size_correct) {
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      std::ostringstream output_ss;
      std::string separator = "";

      for (int i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";
      }

      TORCH_CHECK(
          0,
          "Calculated padded input size per channel: (",
          input_ss.str(),
          "). "
          "Kernel size: (",
          kernel_ss.str(),
          "). Kernel size can't be greater than actual input size");
    }
  } else {
    TORCH_CHECK(
        input.size(1) == weight_sizes[0],
        "Given transposed=",
        transposed,
        ", weight of size ",
        weight_sizes,
        ", expected input",
        input.sizes(),
        " to have ",
        weight_sizes[0],
        " channels, but got ",
        input.size(1),
        " channels instead");
    TORCH_CHECK(
        !bias.defined() ||
            (bias.ndimension() == 1 &&
             bias.size(0) == weight_sizes[1] * groups),
        "Given transposed=",
        transposed,
        ", weight of size ",
        weight_sizes,
        ", expected bias to be 1-dimensional with ",
        weight_sizes[1] * groups,
        " elements",
        ", but got bias of size ",
        bias.sizes(),
        " instead");
  }
}

static at::Tensor view4d(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.ndimension() == 3,
      "expected 3D tensor, got tensor with ",
      tensor.ndimension(),
      " dimensions instead");
  return tensor.unsqueeze(2);
}

static at::Tensor view3d(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.ndimension() == 4,
      "expected 4D tensor, got tensor with ",
      tensor.ndimension(),
      " dimensions instead");
  return tensor.squeeze(2);
}

} // namespace impl

using namespace impl;

Tensor _convolution_out(
    Tensor& output_r,
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    ConvAttr attr) {
  auto output = output_r;
  auto input = input_r.contiguous();
  auto weight = weight_r.contiguous();
  auto bias = bias_r;
  auto k = weight.ndimension();
  if (k == input.ndimension() + 1) {
    k = input.ndimension();
  }
  int64_t dim = k - 2;

  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

  ConvParams params;
  params.stride = expand_param_if_needed(stride_, "stride", dim);
  params.padding = expand_param_if_needed(padding_, "padding", dim);
  params.dilation = expand_param_if_needed(dilation_, "dilation", dim);
  params.transposed = transposed_;
  params.output_padding =
      expand_param_if_needed(output_padding_, "output_padding", dim);
  params.groups = groups_;

  check_shape_forward(input, weight, bias, params, true);

  if (k == 3) {
    params.view1d_as_2d();
    input = view4d(input);
    weight = view4d(weight);
  }

  Tensor output_;

  if(transposed_) {
    output_ = dpcpp_convolution_transpose(
	input,
	weight,
	bias,
	params.stride,
	params.padding,
	params.output_padding,
	params.dilation,
	params.groups);
  } else {
    output_ = convolution(
	output,
	input,
	weight,
	bias,
	params.padding,
	params.stride,
	params.dilation,
	params.groups,
	attr);

    if (k == 3) {
      output_ = view3d(output_);
    }
  }

  return output_;
}

Tensor _convolution(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    ConvAttr attr) {
  Tensor output_r;
  return _convolution_out(
      output_r,
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      transposed_,
      output_padding_,
      groups_,
      attr);
}

Tensor convolution_sum(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    Tensor& accumu,
    Scalar scale,
    Scalar alpha,
    Scalar beta) {
  ConvAttr attr(
      scale.to<float>(),
      alpha.to<float>(),
      beta.to<float>(),
      1.f,
      ConvAttr::kind_with_sum);
  return _convolution_out(
      accumu,
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      transposed_,
      output_padding_,
      groups_,
      attr);
}

Tensor convolution_sum_relu(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    Tensor& accumu,
    Scalar scale,
    Scalar alpha,
    Scalar beta) {
  ConvAttr attr(
      scale.to<float>(),
      alpha.to<float>(),
      beta.to<float>(),
      1.f,
      ConvAttr::kind_with_relu | ConvAttr::kind_with_sum);
  return _convolution_out(
      accumu,
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      transposed_,
      output_padding_,
      groups_,
      attr);
}

Tensor convolution_relu(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    Scalar scale,
    Scalar alpha,
    Scalar beta) {
  ConvAttr attr(
      scale.to<float>(),
      alpha.to<float>(),
      beta.to<float>(),
      1.f,
      ConvAttr::kind_with_relu);
  return _convolution(
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      transposed_,
      output_padding_,
      groups_,
      attr);
}

Tensor convolution_sigmoid(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    Scalar scale,
    Scalar alpha,
    Scalar beta) {
  ConvAttr attr(
      scale.to<float>(),
      alpha.to<float>(),
      beta.to<float>(),
      1.f,
      ConvAttr::kind_with_sigmoid);
  return _convolution(
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      transposed_,
      output_padding_,
      groups_,
      attr);
}

Tensor convolution_overrideable(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_) {
  return _convolution(
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      transposed_,
      output_padding_,
      groups_,
      ConvAttr());
}

std::tuple<Tensor, Tensor, Tensor> convolution_backward_overrideable(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {

  Tensor input_ = input;
  // oneDNN can revice non-contiguous input if we define the stride in input_md,
  // for now, we contiguous the input before oneDNN.
  if (!input.is_contiguous()) {
    input_ = input.contiguous();
  }

  Tensor grad_output_ = grad_output.contiguous();

  Tensor grad_input, grad_weight, grad_bias;

  if (output_mask[0]) {
    if(transposed){
      grad_input = dpcpp_convolution_transpose_backward_input(
	  input,
	  weight,
	  grad_output,
	  stride,
	  padding,
	  dilation,
	  groups,
	  output_mask[2]);
    } else {
      grad_input = dpcpp_convolution_backward_input(
	  input_.sizes(),
	  grad_output_,
	  weight,
	  padding,
	  stride,
	  dilation,
	  groups,
	  output_mask[2]);
    }
  }
  if (output_mask[1] || output_mask[2]) {
    if(transposed){
      std::tie(grad_weight, grad_bias) = dpcpp_convolution_transpose_backward_weights(
	  input,
	  weight,
	  grad_output,
	  stride,
	  padding,
	  dilation,
	  groups,
	  output_mask[2]);
    } else {
      std::tie(grad_weight, grad_bias) = dpcpp_convolution_backward_weights(
	  weight.sizes(),
	  grad_output_,
	  input_,
	  padding,
	  stride,
	  dilation,
	  groups,
	  output_mask[2]);
    }
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

} // namespace AtenIpexTypeXPU
} // namespace at
