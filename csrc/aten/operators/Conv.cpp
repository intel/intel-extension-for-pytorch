#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <core/TensorImplUtils.h>
#include <quantized/Quantizer.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>

#include <oneDNN/oneDNN.h>
#include "ATen/core/interned_strings.h"
#include "c10/util/Optional.h"
#include "comm/ParamUtils.h"
#include "comm/RegistrationDeclarations.h"

using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {
namespace {

/* IPEX_CONV_DEFINATION
This macro is used to generate the defination of conv2d, _convolution related
post-op functions in a convinent way. It can only be used when post-op's name in
function defination is exactly the same as the name in Attr's defined post-ops,
and no any extra parameters is brought in compared to the original convolution
signiture.
*/

#define IPEX_CONV_DEFINATION(op)                                   \
  Tensor convolution_##op(                                         \
      const Tensor& input,                                         \
      const Tensor& weight,                                        \
      const c10::optional<Tensor>& bias,                           \
      std::vector<int64_t> stride_,                                \
      std::vector<int64_t> padding_,                               \
      std::vector<int64_t> dilation_,                              \
      int64_t groups_) {                                           \
    Attr att;                                                      \
    att.append_post_eltwise(1.0f, 0.0f, 0.0f, att.kind_with_##op); \
    Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor(); \
    return _convolution(                                           \
        input,                                                     \
        weight,                                                    \
        bias_,                                                     \
        stride_,                                                   \
        padding_,                                                  \
        dilation_,                                                 \
        false,                                                     \
        {{0, 0}},                                                  \
        groups_,                                                   \
        att);                                                      \
  }                                                                \
                                                                   \
  Tensor _convolution_##op(                                        \
      const Tensor& input,                                         \
      const Tensor& weight,                                        \
      const c10::optional<Tensor>& bias,                           \
      std::vector<int64_t> stride_,                                \
      std::vector<int64_t> padding_,                               \
      std::vector<int64_t> dilation_,                              \
      bool transposed,                                             \
      std::vector<int64_t> output_padding_,                        \
      int groups,                                                  \
      bool benchmark,                                              \
      bool deterministic,                                          \
      bool cudnn_enabled,                                          \
      bool allow_tf32) {                                           \
    Attr att;                                                      \
    att.append_post_eltwise(1.0f, 0.0f, 0.0f, att.kind_with_##op); \
    Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor(); \
    return _convolution(                                           \
        input,                                                     \
        weight,                                                    \
        bias_,                                                     \
        stride_,                                                   \
        padding_,                                                  \
        dilation_,                                                 \
        transposed,                                                \
        output_padding_,                                           \
        groups,                                                    \
        att);                                                      \
  }
} // namespace
namespace impl {

Tensor dpcpp_convolution_backward_input(
    IntArrayRef input_size,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined) {
  bool conv_double = grad_output.scalar_type() == kDouble;
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto ndim = grad_output.ndimension();
  TORCH_CHECK(
      3 == ndim || 4 == ndim || 5 == ndim,
      "convolution bwd input only supports 3D, 4D, 5D tensor");
  auto gy_cl_tag = get_cl_tag_by_ndim(ndim);
  // smf: suggest memory format
  auto smf = onednn_conv_use_channels_last(grad_output, weight)
      ? gy_cl_tag
      : at::MemoryFormat::Contiguous;
  auto grad_input = at::empty(input_size, grad_output.options(), smf);

  if (grad_input.numel() == 0) {
    return grad_input;
  }

  auto ic = grad_input.size(1);
  auto oc = grad_output.size(1);

  // align data type with bf16
  auto data_grad = get_onednn_dtype_include_double(grad_output);
  auto weight_t = get_onednn_dtype_include_double(weight);
  auto bias_t =
      conv_double ? dnnl::memory::data_type::f64 : dnnl::memory::data_type::f32;
  auto weight_usr_t = weight_t;
  auto format_any = memory::format_tag::any;
  auto format_input =
      conv_src_fmt(ndim, onednn_conv_use_channels_last(grad_output, weight));
  auto format_weight = conv_wgh_fmt(
      ndim, groups != 1, onednn_conv_use_channels_last(grad_output, weight));

  memory::dims input_tz = grad_input.sizes().vec();
  memory::dims weight_tz =
      compatible_wgh_dims(ndim, groups, oc, ic, weight.sizes());
  memory::dims bias_tz = {oc};
  memory::dims output_tz = grad_output.sizes().vec();
  output_tz[0] = grad_input.size(0); // set n

  memory::dims _stride = stride.vec();
  memory::dims _padding_front_top_left = padding_front_top_left.vec();
  memory::dims _padding_back_bottom_right = padding_back_bottom_right.vec();
  memory::dims _dilation = compatible_dilation(dilation);

  // Master weight
  if (data_grad == dnnl::memory::data_type::bf16) {
    weight_t = dnnl::memory::data_type::bf16;
    bias_t = dnnl::memory::data_type::bf16;
  }

  auto input_md = Settings::I().is_onednn_layout_enabled()
      ? memory::desc(input_tz, data_grad, format_any)
      : memory::desc(input_tz, data_grad, format_input);
  auto weight_md = (onednn_conv_use_channels_last(grad_output, weight) ||
                    Settings::I().is_onednn_layout_enabled())
      ? memory::desc(weight_tz, weight_t, format_any)
      : memory::desc(weight_tz, weight_t, format_weight);
  auto output_md = Settings::I().is_onednn_layout_enabled()
      ? memory::desc(output_tz, data_grad, format_any)
      : memory::desc(output_tz, data_grad, format_input);
  auto bias_md =
      bias_defined ? memory::desc(bias_tz, bias_t, format_any) : memory::desc();

  primitive_attr pattr;
  if (data_grad == memory::data_type::f32) {
    pattr.set_fpmath_mode(xpu::oneDNN::get_onednn_fpmath_mode());
  }

#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

  auto conv_forward_desc = convolution_forward::desc(
      prop_kind::forward,
      algorithm::convolution_direct,
      input_md,
      weight_md,
      bias_md,
      output_md,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right);

  auto conv_forward_pd =
      convolution_forward::primitive_desc(conv_forward_desc, pattr, engine);

  auto conv_backward_data_desc = convolution_backward_data::desc(
      algorithm::convolution_direct,
      input_md,
      weight_md,
      output_md,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right);

  auto conv_backward_data_pd = convolution_backward_data::primitive_desc(
      conv_backward_data_desc, pattr, engine, conv_forward_pd);

  memory grad_output_usr_memory, weight_usr_memory, grad_input_usr_memory;
  if (!Settings::I().is_onednn_layout_enabled()) {
    grad_output_usr_memory = dpcpp_onednn_memory(
        {output_tz, data_grad, format_input}, engine, grad_output.data_ptr());

    weight_usr_memory = dpcpp_onednn_memory(
        {weight_tz, weight_usr_t, format_weight}, engine, weight.data_ptr());

    grad_input_usr_memory = dpcpp_onednn_memory(
        {input_tz, data_grad, format_input}, engine, grad_input.data_ptr());
  } else {
    auto grad_output_ctx =
        at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(grad_output);
    grad_output_usr_memory = grad_output_ctx.is_plain()
        ? dpcpp_onednn_memory(
              {output_tz, data_grad, format_input},
              engine,
              grad_output.data_ptr())
        : dpcpp_onednn_memory(
              {grad_output_ctx.meta()}, engine, grad_output.data_ptr());

    auto weight_ctx =
        at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(weight);
    weight_usr_memory = weight_ctx.is_plain()
        ? dpcpp_onednn_memory(
              {weight_tz, weight_usr_t, format_weight},
              engine,
              weight.data_ptr())
        : dpcpp_onednn_memory({weight_ctx.meta()}, engine, weight.data_ptr());

    auto grad_input_ctx =
        at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(grad_input);
    grad_input_usr_memory = grad_input_ctx.is_plain()
        ? dpcpp_onednn_memory(
              {output_tz, data_grad, format_input},
              engine,
              grad_input.data_ptr())
        : dpcpp_onednn_memory(
              {grad_input_ctx.meta()}, engine, grad_input.data_ptr());
  }

  Tensor grad_output_;
  auto expected_grad_output_md = conv_backward_data_pd.diff_dst_desc();
  auto grad_output_memory = grad_output_usr_memory;
  if (grad_output_usr_memory.get_desc() != expected_grad_output_md) {
    grad_output_ = empty_opaque_tensor(
        expected_grad_output_md, grad_output.options(), c10::nullopt);
    grad_output_memory = dpcpp_onednn_memory(
        expected_grad_output_md, engine, grad_output_.data_ptr());
    xpu::oneDNN::reorder(grad_output, grad_output_);
  }

  Tensor weight_;
  auto expected_weight_md = conv_backward_data_pd.weights_desc();
  auto weight_memory = weight_usr_memory;
  if (weight_usr_memory.get_desc() != expected_weight_md) {
    weight_ =
        empty_opaque_tensor(expected_weight_md, weight.options(), c10::nullopt);
    weight_memory =
        dpcpp_onednn_memory(expected_weight_md, engine, weight_.data_ptr());
    xpu::oneDNN::reorder(weight, weight_);
  }

  Tensor grad_input_;
  auto expected_grad_input_md = conv_backward_data_pd.diff_src_desc();
  auto grad_input_memory = grad_input_usr_memory;
  if (grad_input_memory.get_desc() != expected_grad_input_md) {
    grad_input_ =
        empty_opaque_tensor(expected_grad_input_md, grad_input.options(), smf);
    grad_input_memory = dpcpp_onednn_memory(
        expected_grad_input_md, engine, grad_input_.data_ptr());
  }

#ifdef USE_SCRATCHPAD_MODE
  int scratchpad_size = conv_backward_data_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, grad_output.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_memory = dpcpp_onednn_memory(
      conv_backward_data_pd.scratchpad_desc(),
      engine,
      scratchpad_tensor.data_ptr());
#endif

  auto conv_backward_data = convolution_backward_data(conv_backward_data_pd);
  DPCPP_ONEDNN_EXEC(
      conv_backward_data,
      strm,
      {
          {DNNL_ARG_DIFF_DST, grad_output_memory},
          {DNNL_ARG_WEIGHTS, weight_memory},
          {DNNL_ARG_DIFF_SRC, grad_input_memory},
#ifdef USE_SCRATCHPAD_MODE
          {DNNL_ARG_SCRATCHPAD, scratchpad_memory},
#endif
      });

  if (!Settings::I().is_onednn_layout_enabled() &&
      grad_input_memory != grad_input_usr_memory) {
    xpu::oneDNN::reorder(grad_input_, grad_input);
  } else if (
      Settings::I().is_onednn_layout_enabled() &&
      grad_input_memory != grad_input_usr_memory) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(grad_input_);
    DPCPPTensorContext::set_tensor_ctx(grad_input, std::move(blk_ctx));
  }

  return grad_input;
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

bool ConvParams::is_depthwise(const at::Tensor& input, const at::Tensor& weight)
    const {
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
    Attr attr,
    IntArrayRef pad_nd = IntArrayRef({})) {
  auto output = output_r;
  auto ndim = input_r.ndimension();
  TORCH_CHECK(
      3 == ndim || 4 == ndim || 5 == ndim,
      "convolution only supports 3D, 4D, 5D tensor");
  auto mem_fmt = get_cl_tag_by_ndim(ndim);
  auto input = onednn_conv_use_channels_last(input_r, weight_r)
      ? input_r.contiguous(mem_fmt)
      : input_r.contiguous();
  auto weight = onednn_conv_use_channels_last(input_r, weight_r)
      ? weight_r.contiguous(mem_fmt)
      : weight_r.contiguous();
  auto bias = bias_r.contiguous();
  auto k = weight.ndimension();
  if (k == input.ndimension() + 1) {
    k = input.ndimension();
  }
  int64_t dim = k - 2;

  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

  ConvParams params;
  params.stride = expand_param_if_needed(stride_, "stride", dim);
  // PyTorch default Conv padding should be a single integer value
  // or a list of values to match the conv dimensions
  // conv2d, the number of padding values should be 1 or 2
  // conv3d, the number of padding values should be 1 or 3
  // the padding value will be padded into both side of Conv input (D, H, W)
  params.padding = expand_param_if_needed(padding_, "padding", dim);

  // oneDNN supports padding the two sides of src with different values
  // the padding order should be front_top_left and back_bottom_right
  auto padding_front_top_left = params.padding;
  auto padding_back_bottom_right = params.padding;

  // PyTorch constant_pad_nd:
  // can pad different value to the two sides of Conv input (W, H, D)
  // (padding_left, padding_right,
  //  padding_top, padding_bottom,
  //  padding_front, padding_back)
  if (pad_nd.vec().size() > 0) {
    for (int i = 0; i < dim; ++i) {
      padding_front_top_left[i] += pad_nd[2 * dim - 2 * i - 2]; // 4, 2, 0
      padding_back_bottom_right[i] += pad_nd[2 * dim - 2 * i - 1]; // 5, 3, 1
    }
  }

  params.dilation = expand_param_if_needed(dilation_, "dilation", dim);
  params.transposed = transposed_;
  params.output_padding =
      expand_param_if_needed(output_padding_, "output_padding", dim);
  params.groups = groups_;

  check_shape_forward(input, weight, bias, params, true);

  Tensor output_;

  if (transposed_) {
    // TODO::
    // currently only support deconvolution padding with the same value on two
    // side need to check deconvolution support different padding values or not
    output_ = xpu::oneDNN::deconvolution(
        input,
        weight,
        bias,
        params.stride,
        params.padding,
        params.output_padding,
        params.dilation,
        params.groups);
  } else {
    output_ = xpu::oneDNN::convolution(
        output,
        input,
        weight,
        bias,
        padding_front_top_left,
        padding_back_bottom_right,
        params.stride,
        params.dilation,
        params.groups,
        attr);
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
    Attr attr) {
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

Tensor pad_convolution(
    const Tensor& input_r,
    IntArrayRef pad_nd,
    Scalar value,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_) {
  // oneDNN only support padding value with 0
  if (value.to<float>() != 0.0) {
    auto padded_input = at::constant_pad_nd(input_r, pad_nd, value);
    Tensor output_r;
    return _convolution_out(
        output_r,
        padded_input,
        weight_r,
        bias_r,
        stride_,
        padding_,
        dilation_,
        false,
        {{0, 0}},
        groups_,
        Attr());
  }

  Tensor output_r;
  return _convolution_out(
      output_r,
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      Attr(),
      pad_nd);
}

Tensor convolution_overrideable(
    const Tensor& input_r,
    const Tensor& weight_r,
    const c10::optional<at::Tensor>& bias_r_opt,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_) {
  c10::MaybeOwned<Tensor> bias_r_maybe_owned =
      at::borrow_from_optional_tensor(bias_r_opt);
  const Tensor& bias_r = *bias_r_maybe_owned;
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
      Attr());
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
  // Tensor input_ = input;
  // oneDNN can revice non-contiguous input if we define the stride in input_md,
  // for now, we contiguous the input before oneDNN.
  auto input_ndim = input.ndimension();
  TORCH_CHECK(
      3 == input_ndim || 4 == input_ndim || 5 == input_ndim,
      "convolution bwd only supports 3D, 4D, 5D tensor");

  TORCH_CHECK(
      grad_output.scalar_type() == ScalarType::Float ||
          grad_output.scalar_type() == ScalarType::BFloat16 ||
          grad_output.scalar_type() == ScalarType::Double,
      "so far only support float, bfloat16 and double convolution backward in XPU backend, your data type is ",
      grad_output.scalar_type());

  auto cl_tag = get_cl_tag_by_ndim(input_ndim);
  // try best to make sure that suggest memory format is propagated
  Tensor input_ = (grad_output.suggest_memory_format() == cl_tag ||
                   input.suggest_memory_format() == cl_tag ||
                   weight.suggest_memory_format() == cl_tag)
      ? input.contiguous(cl_tag)
      : input.contiguous();

  Tensor grad_output_ = (grad_output.suggest_memory_format() == cl_tag ||
                         input.suggest_memory_format() == cl_tag ||
                         weight.suggest_memory_format() == cl_tag)
      ? grad_output.contiguous(cl_tag)
      : grad_output.contiguous();

  Tensor grad_input, grad_weight, grad_bias;

  if (output_mask[0]) {
    if (transposed) {
      grad_input = xpu::oneDNN::deconvolution_backward_data(
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
          padding,
          stride,
          dilation,
          groups,
          output_mask[2]);
    }
  }
  if (output_mask[1] || output_mask[2]) {
    if (transposed) {
      std::tie(grad_weight, grad_bias) =
          xpu::oneDNN::deconvolution_backward_weights(
              input,
              weight,
              grad_output,
              stride,
              padding,
              dilation,
              groups,
              output_mask[2]);
    } else {
      std::tie(grad_weight, grad_bias) =
          xpu::oneDNN::convolution_backward_weights(
              grad_output_,
              input_,
              weight.sizes(),
              padding,
              padding,
              stride,
              dilation,
              groups,
              output_mask[2]);
    }
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

// It is recommand to define the post-op function name follow the pytorch's
// inner op rather than oneDNN post op name, for example, we can find pytorch's
// silu op named swish in oneDNN. For clarity, we use all the pytorch's op name
// in function defination. Therefore, for the fusion of convolution + silu, we
// name the funciton as convolution_silu.
IPEX_CONV_DEFINATION(sqrt)
IPEX_CONV_DEFINATION(abs)
IPEX_CONV_DEFINATION(tanh)
IPEX_CONV_DEFINATION(square)
IPEX_CONV_DEFINATION(exp)
IPEX_CONV_DEFINATION(log)
IPEX_CONV_DEFINATION(round)
IPEX_CONV_DEFINATION(sigmoid)
IPEX_CONV_DEFINATION(relu)
IPEX_CONV_DEFINATION(hardswish)
IPEX_CONV_DEFINATION(mish)
IPEX_CONV_DEFINATION(gelu)

Tensor convolution_silu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_) {
  Attr att;
  att.append_post_eltwise(1.0f, 1.0f, 0.0f, att.kind_with_swish);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_silu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {
  Attr att;
  att.append_post_eltwise(1.0f, 1.0f, 0.0f, att.kind_with_swish);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}

Tensor convolution_log_sigmoid(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_) {
  Attr att;
  att.append_post_eltwise(1.0f, 1.0f, 0.0f, att.kind_with_logsigmoid);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_log_sigmoid(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {
  Attr att;
  att.append_post_eltwise(1.0f, 1.0f, 0.0f, att.kind_with_logsigmoid);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}

Tensor convolution_hardsigmoid(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_) {
  Attr att;
  att.append_post_eltwise(
      1.0f, 1.0f / 6., 1.0f / 2., att.kind_with_hardsigmoid);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_hardsigmoid(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {
  Attr att;
  att.append_post_eltwise(
      1.0f, 1.0f / 6., 1.0f / 2., att.kind_with_hardsigmoid);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}

Tensor convolution_pow(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_,
    Scalar exponent) {
  Attr att;
  att.append_post_eltwise(1.0f, 1.0f, exponent.toFloat(), att.kind_with_pow);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_pow(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Scalar exponent) {
  Attr att;
  att.append_post_eltwise(1.0f, 1.0f, exponent.toFloat(), att.kind_with_pow);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}

Tensor convolution_leaky_relu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_,
    Scalar negative_slope) {
  Attr att;
  att.append_post_eltwise(
      1.0f, negative_slope.toFloat(), 0.f, att.kind_with_relu);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_leaky_relu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Scalar negative_slope) {
  Attr att;
  att.append_post_eltwise(
      1.0f, negative_slope.toFloat(), 0.f, att.kind_with_relu);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}

Tensor convolution_hardtanh(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_,
    Scalar minval,
    Scalar maxval) {
  Attr att;
  att.append_post_eltwise(
      1.0f, minval.toFloat(), maxval.toFloat(), att.kind_with_clip);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_hardtanh(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Scalar minval,
    Scalar maxval) {
  Attr att;
  att.append_post_eltwise(
      1.0f, minval.toFloat(), maxval.toFloat(), att.kind_with_clip);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}

Tensor convolution_elu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale) {
  AT_ASSERT(
      scale.toFloat() == 1.0f && input_scale.toFloat() == 1.0f,
      "elu's scale and input scale can only be 1.f in jit fusion");
  Attr att;
  att.append_post_eltwise(1.0f, alpha.toFloat(), 1.0f, att.kind_with_elu);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      att);
}

Tensor _convolution_elu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale) {
  AT_ASSERT(
      scale.toFloat() == 1.0f && input_scale.toFloat() == 1.0f,
      "elu's scale and input scale can only be 1.f in jit fusion");
  Attr att;
  att.append_post_eltwise(1.0f, alpha.toFloat(), 1.0f, att.kind_with_elu);
  Tensor bias_ = bias.has_value() ? bias.value() : at::Tensor();
  return _convolution(
      input,
      weight,
      bias_,
      stride_,
      padding_,
      dilation_,
      transposed,
      output_padding_,
      groups,
      att);
}

Tensor convolution_sum(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_,
    Tensor& accumu,
    Scalar scale) {
  // only support scale = 1.0f in oneDNN for non-quantized case.
  TORCH_CHECK(
      scale.to<float>() == 1.f,
      "only support convolution sum fusion with sum scale equals to 1");
  // output = conv_scale * (Conv(input, weight) + bias) + sum_scale * accumu;
  Attr attr;
  attr.append_post_sum(/* sum_scale */ scale.to<float>()); // append post op sum
  Tensor res = _convolution_out(
      accumu,
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      attr);
  return res;
}

Tensor _convolution_sum(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Tensor& accumu,
    Scalar scale) {
  // only support scale = 1.0f in oneDNN for non-quantized case.
  TORCH_CHECK(
      scale.to<float>() == 1.f,
      "only support convolution sum fusion with sum scale equals to 1");
  // output = conv_scale * (Conv(input, weight) + bias) + sum_scale * accumu;
  Attr attr;
  attr.append_post_sum(/* sum_scale */ scale.to<float>()); // append post op sum
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
    int64_t groups_,
    Tensor& accumu,
    Scalar scale) {
  // only support scale = 1.0f in oneDNN for non-quantized case.
  TORCH_CHECK(
      scale.to<float>() == 1.f,
      "only support convolution sum fusion with sum scale equals to 1");
  // output = relu_scale * Relu(conv_scale * (Conv(input, weight) + bias) +
  // sum_scale * accumu);
  Attr attr;
  attr.append_post_sum(/* sum_scale */ scale.to<float>()); // append post op sum
  attr.append_post_eltwise( // append post relu
      /* relu_scale */ 1.f,
      /* alpha */ 0.f,
      /* beta */ 0.f,
      attr.kind_with_relu);
  return _convolution_out(
      accumu,
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      attr);
}

Tensor _convolution_sum_relu(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Tensor& accumu,
    Scalar scale) {
  // only support scale = 1.0f in oneDNN for non-quantized case.
  TORCH_CHECK(
      scale.to<float>() == 1.f,
      "only support convolution sum fusion with sum scale equals to 1");
  // output = relu_scale * Relu(conv_scale * (Conv(input, weight) + bias) +
  // sum_scale * accumu);
  Attr attr;
  attr.append_post_sum(/* sum_scale */ scale.to<float>()); // append post op sum
  attr.append_post_eltwise( // append post relu
      /* relu_scale */ 1.f,
      /* alpha */ 0.f,
      /* beta */ 0.f,
      attr.kind_with_relu);
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

Tensor convolution_binary_mul(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_,
    const Tensor& binary) {
  Attr attr;
  attr.append_post_binary(attr.kind_with_binary_mul, binary);
  Tensor output_r;
  return _convolution_out(
      output_r,
      input_r,
      weight_r,
      bias_r,
      stride_,
      padding_,
      dilation_,
      false,
      {{0, 0}},
      groups_,
      attr);
}

} // namespace AtenIpexTypeXPU
} // namespace at
