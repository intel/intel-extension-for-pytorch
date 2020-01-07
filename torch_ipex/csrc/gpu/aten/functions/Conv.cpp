#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#include <core/SYCLUtils.h>
#include <core/Runtime.h>

#include <utils/ParamUtils.h>


using namespace mkldnn;

namespace at { namespace native {

constexpr int input_batch_size_dim = 0;  // also grad_input
constexpr int weight_output_channels_dim = 0;

static std::vector<int64_t> conv_output_size(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
{
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2])
                        - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

at::Tensor sycl_convolution(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
{
  auto output = at::empty(conv_output_size(
    input.sizes(), weight.sizes(), padding, stride, dilation, groups), input.options());

  Device curDevice = Device(kDPCPP, c10::sycl::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  auto strm = GpuStreamManager::Instance().get_stream();

  int32_t g = groups;

  int32_t n = input.size(0);
  int32_t ic = input.size(1);
  int32_t ih = input.size(2);
  int32_t iw = input.size(3);

  int32_t oc = output.size(1);
  int32_t oh = output.size(2);
  int32_t ow = output.size(3);

  int32_t kh = weight.size(2);
  int32_t kw = weight.size(3);

  int32_t sh = stride[0];
  int32_t sw = stride[1];
  int32_t ph = padding[0];
  int32_t pw = padding[1];

  auto data_t = dt_to_dnnl(input.type().scalarType());
  auto format_any = memory::format_tag::any;
  auto format_nchw = memory::format_tag::nchw;
  auto format_weight = (g!= 1) ? memory::format_tag::goihw : memory::format_tag::oihw;
  auto format_x = memory::format_tag::x;

  memory::dims input_tz = {n, ic, ih, iw};
  memory::dims weight_tz = (g!= 1) ? memory::dims{g, oc/g, ic/g, kh, kw} : memory::dims{oc, ic, kh, kw};
  memory::dims bias_tz = {oc};
  memory::dims output_tz = {n, oc, oh, ow};
  memory::dims _stride = {sh, sw};
  memory::dims _padding = {ph, pw};

  auto input_md = memory::desc({input_tz}, data_t, format_any);
  auto weight_md = memory::desc({weight_tz}, data_t, format_any);
  auto bias_md = memory::desc({bias_tz}, data_t, format_any);
  auto output_md = memory::desc({output_tz}, data_t, format_any);

  std::shared_ptr<convolution_forward::desc> conv_forward_desc;
  if (bias.defined()) {
    conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
      algorithm::convolution_direct, input_md, weight_md, bias_md, output_md,
      _stride, _padding, _padding));
  } else {
    conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
      algorithm::convolution_direct, input_md, weight_md, output_md,
      _stride, _padding, _padding));
  }

  std::shared_ptr<convolution_forward::primitive_desc> conv_forward_pd;
  conv_forward_pd.reset(new convolution_forward::primitive_desc(
    *conv_forward_desc, engine));

  auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(input.data_ptr(), input_usr_memory);

  auto weight_usr_memory = memory({{{weight_tz}, data_t,  format_weight}, engine});
  sycl_set_mkldnn_buffer(weight.data_ptr(), weight_usr_memory);

  auto output_usr_memory = memory({{{output_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(output.data_ptr(), output_usr_memory);

  auto expected_input_md = conv_forward_pd->src_desc();
  auto input_memory = input_usr_memory;
  if (input_usr_memory.get_desc() != expected_input_md) {
    input_memory = memory(expected_input_md, engine);
    reorder(input_usr_memory, input_memory).
        execute(strm, input_usr_memory, input_memory);
  }

  auto expected_weight_md = conv_forward_pd->weights_desc();
  auto weight_memory = weight_usr_memory;
  if (weight_usr_memory.get_desc() != expected_weight_md) {
    weight_memory = memory(expected_weight_md, engine);
    reorder(weight_usr_memory, weight_memory).
        execute(strm, weight_usr_memory, weight_memory);
  }

  auto expected_output_md = conv_forward_pd->dst_desc();
  auto output_memory = output_usr_memory;
  if (output_usr_memory.get_desc() != expected_output_md) {
    output_memory = memory(expected_output_md, engine);
  }

  std::shared_ptr<convolution_forward> conv_forward;
  std::shared_ptr<memory> bias_usr_memory;
  if (bias.defined()) {
    bias_usr_memory.reset(new memory({{{bias_tz}, data_t, format_x}, engine}));
    sycl_set_mkldnn_buffer(bias.data_ptr(), *bias_usr_memory);
  } else {
    bias_usr_memory.reset(new memory({{{}, data_t, format_x}, engine}));
  }

  conv_forward.reset(new convolution_forward(*conv_forward_pd));
  conv_forward->execute(strm, {
      {MKLDNN_ARG_SRC, input_memory},
      {MKLDNN_ARG_WEIGHTS, weight_memory},
      {MKLDNN_ARG_BIAS, *bias_usr_memory},
      {MKLDNN_ARG_DST, output_memory}});

  if (output_memory != output_usr_memory) {
    reorder(output_memory, output_usr_memory).
        execute(strm, output_memory, output_usr_memory);
  }

  return output;
}

Tensor sycl_convolution_backward_input(
    IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  auto grad_input = at::empty(input_size, grad_output.options());

  Device curDevice = Device(kDPCPP, c10::sycl::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  auto strm = GpuStreamManager::Instance().get_stream();

  int32_t g = groups;

  int32_t n = grad_input.size(0);
  int32_t ic = grad_input.size(1);
  int32_t ih = grad_input.size(2);
  int32_t iw = grad_input.size(3);

  int32_t oc = grad_output.size(1);
  int32_t oh = grad_output.size(2);
  int32_t ow = grad_output.size(3);

  int32_t kh = weight.size(2);
  int32_t kw = weight.size(3);

  int32_t sh = stride[0];
  int32_t sw = stride[1];
  int32_t ph = padding[0];
  int32_t pw = padding[1];

  auto data_t = memory::data_type::f32;
  auto format_any = memory::format_tag::any;
  auto format_nchw = memory::format_tag::nchw;
  auto format_weight = (g!= 1) ? memory::format_tag::goihw : memory::format_tag::oihw;

  memory::dims input_tz = {n, ic, ih, iw};
  memory::dims weight_tz = (g!= 1) ? memory::dims{g, oc/g, ic/g, kh, kw} : memory::dims{oc, ic, kh, kw};
  memory::dims bias_tz = {oc};
  memory::dims output_tz = {n, oc, oh, ow};
  memory::dims _stride = {sh, sw};
  memory::dims _padding = {ph, pw};

  auto input_md = memory::desc({input_tz}, data_t, format_any);
  auto weight_md = memory::desc({weight_tz}, data_t, format_any);
  auto bias_md = memory::desc({bias_tz}, data_t, format_any);
  auto output_md = memory::desc({output_tz}, data_t, format_any);

  // need to re-create conv_forward_pd to feed conv_backward_data_pd
  std::shared_ptr<convolution_forward::desc> conv_forward_desc;
  if (bias_defined) {
    conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
      algorithm::convolution_direct, input_md, weight_md, bias_md, output_md,
      _stride, _padding, _padding));
  } else {
    conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
      algorithm::convolution_direct, input_md, weight_md, output_md,
      _stride, _padding, _padding));
  }

  std::shared_ptr<convolution_forward::primitive_desc> conv_forward_pd;
  conv_forward_pd.reset(new convolution_forward::primitive_desc(
    *conv_forward_desc, engine));

  std::shared_ptr<convolution_backward_data::desc> conv_backward_data_desc;
  conv_backward_data_desc.reset(new convolution_backward_data::desc(
    algorithm::convolution_direct, input_md, weight_md, output_md,
    _stride, _padding, _padding));

  std::shared_ptr<convolution_backward_data::primitive_desc> conv_backward_data_pd;
  conv_backward_data_pd.reset(new convolution_backward_data::primitive_desc(
    *conv_backward_data_desc, engine, *conv_forward_pd));

  auto grad_output_usr_memory = memory({{{output_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(grad_output.data_ptr(), grad_output_usr_memory);

  auto weight_usr_memory = memory({{{weight_tz}, data_t, format_weight}, engine});
  sycl_set_mkldnn_buffer(weight.data_ptr(), weight_usr_memory);

  auto grad_input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(grad_input.data_ptr(), grad_input_usr_memory);

  auto expected_grad_output_md = conv_backward_data_pd->diff_dst_desc();
  auto grad_output_memory = grad_output_usr_memory;
  if (grad_output_usr_memory.get_desc() != expected_grad_output_md) {
    grad_output_memory = memory(expected_grad_output_md, engine);
    reorder(grad_output_usr_memory, grad_output_memory).
        execute(strm, grad_output_usr_memory, grad_output_memory);
  }

  auto expected_weight_md = conv_backward_data_pd->weights_desc();
  auto weight_memory = weight_usr_memory;
  if (weight_usr_memory.get_desc() != expected_weight_md) {
    weight_memory = memory(expected_weight_md, engine);
    reorder(weight_usr_memory, weight_memory).
        execute(strm, weight_usr_memory, weight_memory);
  }

  auto expected_grad_input_md = conv_backward_data_pd->diff_src_desc();
  auto grad_input_memory = grad_input_usr_memory;
  if (grad_input_memory.get_desc() != expected_grad_input_md) {
    grad_input_memory = memory(expected_grad_input_md, engine);
  }

  std::shared_ptr<convolution_backward_data> conv_backward_data;
  conv_backward_data.reset(new convolution_backward_data(*conv_backward_data_pd));
  conv_backward_data->execute(strm, {
     {MKLDNN_ARG_DIFF_DST, grad_output_memory},
     {MKLDNN_ARG_WEIGHTS, weight_memory},
     {MKLDNN_ARG_DIFF_SRC, grad_input_memory}});

  if (grad_input_memory != grad_input_usr_memory) {
    reorder(grad_input_memory, grad_input_usr_memory).
        execute(strm, grad_input_memory, grad_input_usr_memory);
  }

  return grad_input;
}

std::tuple<at::Tensor, at::Tensor> sycl_convolution_backward_weights(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  auto grad_weight = at::empty(weight_size, grad_output.options());

  Tensor grad_bias;
  if (bias_defined) {
    grad_bias = at::empty({grad_output.size(1)}, grad_output.options());
  }

  Device curDevice = Device(kDPCPP, c10::sycl::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  auto strm = GpuStreamManager::Instance().get_stream();

  int32_t g = groups;

  int32_t n = input.size(0);
  int32_t ic = input.size(1);
  int32_t ih = input.size(2);
  int32_t iw = input.size(3);

  int32_t oc = grad_output.size(1);
  int32_t oh = grad_output.size(2);
  int32_t ow = grad_output.size(3);

  int32_t kh = grad_weight.size(2);
  int32_t kw = grad_weight.size(3);

  int32_t sh = stride[0];
  int32_t sw = stride[1];
  int32_t ph = padding[0];
  int32_t pw = padding[1];

  auto data_t = memory::data_type::f32;
  auto format_any = memory::format_tag::any;
  auto format_nchw = memory::format_tag::nchw;
  auto format_weight = (g!= 1) ? memory::format_tag::goihw : memory::format_tag::oihw;
  auto format_x = memory::format_tag::x;

  memory::dims input_tz = {n, ic, ih, iw};
  memory::dims weight_tz = (g!= 1) ? memory::dims{g, oc/g, ic/g, kh, kw} : memory::dims{oc, ic, kh, kw};
  memory::dims bias_tz = {oc};
  memory::dims output_tz = {n, oc, oh, ow};
  memory::dims _stride = {sh, sw};
  memory::dims _padding = {ph, pw};

  memory::desc input_md({input_tz}, data_t, format_any);
  memory::desc weight_md({weight_tz}, data_t, format_any);
  memory::desc bias_md({bias_tz}, data_t, format_any);
  memory::desc output_md({output_tz}, data_t, format_any);

  // need to re-create conv_forward_pd to feed conv_backward_weight_pd
  std::shared_ptr<convolution_forward::desc> conv_forward_desc;
  if (bias_defined) {
    conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
      algorithm::convolution_direct, input_md, weight_md, bias_md, output_md,
      _stride, _padding, _padding));
  } else {
    conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
      algorithm::convolution_direct, input_md, weight_md, output_md,
      _stride, _padding, _padding));
  }

  std::shared_ptr<convolution_forward::primitive_desc> conv_forward_pd;
  conv_forward_pd.reset(new convolution_forward::primitive_desc(
    *conv_forward_desc, engine));

  std::shared_ptr<convolution_backward_weights::desc> conv_backward_weight_desc;
  if (bias_defined) {
    conv_backward_weight_desc.reset(new convolution_backward_weights::desc(
      algorithm::convolution_direct, input_md, weight_md, bias_md, output_md,
      _stride, _padding, _padding));
  } else {
    conv_backward_weight_desc.reset(new convolution_backward_weights::desc(
      algorithm::convolution_direct, input_md, weight_md, output_md,
      _stride, _padding, _padding));
  }

  std::shared_ptr<convolution_backward_weights::primitive_desc> conv_backward_weight_pd;
  conv_backward_weight_pd.reset(new convolution_backward_weights::primitive_desc(
    *conv_backward_weight_desc, engine, *conv_forward_pd));

  auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(input.data_ptr(), input_usr_memory);

  auto grad_output_usr_memory = memory({{{output_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(grad_output.data_ptr(), grad_output_usr_memory);

  auto grad_weight_usr_memory = memory({{{weight_tz}, data_t, format_weight}, engine});
  sycl_set_mkldnn_buffer(grad_weight.data_ptr(), grad_weight_usr_memory);

  std::shared_ptr<memory> grad_bias_memory;

  auto expected_input_md = conv_backward_weight_pd->src_desc();
  auto input_memory = input_usr_memory;
  if (input_usr_memory.get_desc() != expected_input_md) {
    input_memory = memory(expected_input_md, engine);
    reorder(input_usr_memory, input_memory).
        execute(strm, input_usr_memory, input_memory);
  }

  auto expected_grad_output_md = conv_backward_weight_pd->diff_dst_desc();
  auto grad_output_memory = grad_output_usr_memory;
  if (grad_output_usr_memory.get_desc() != expected_grad_output_md) {
    grad_output_memory = memory(expected_grad_output_md, engine);
    reorder(grad_output_usr_memory, grad_output_memory).
        execute(strm, grad_output_usr_memory, grad_output_memory);
  }

  auto expected_grad_weight_md = conv_backward_weight_pd->diff_weights_desc();
  auto grad_weight_memory = grad_weight_usr_memory;
  if (grad_weight_usr_memory.get_desc() != expected_grad_weight_md) {
    grad_weight_memory = memory(expected_grad_weight_md, engine);
  }

  std::shared_ptr<convolution_backward_weights> conv_backward_weight;
  if (bias_defined) {
    grad_bias_memory.reset(new memory({{{bias_tz}, data_t, format_x}, engine}));
    sycl_set_mkldnn_buffer(grad_bias.data_ptr(), *grad_bias_memory);
  } else {
    grad_bias_memory.reset(new memory({{{}, data_t, format_x}, engine}));
  }

  conv_backward_weight.reset(
      new convolution_backward_weights(*conv_backward_weight_pd));
  conv_backward_weight->execute(strm, {
      {MKLDNN_ARG_DIFF_DST, grad_output_memory},
      {MKLDNN_ARG_SRC, input_memory},
      {MKLDNN_ARG_DIFF_WEIGHTS, grad_weight_memory},
      {MKLDNN_ARG_DIFF_BIAS, *grad_bias_memory}});

  if (grad_weight_memory != grad_weight_usr_memory) {
    reorder(grad_weight_memory, grad_weight_usr_memory).
        execute(strm, grad_weight_memory, grad_weight_usr_memory);
  }

  return std::tuple<at::Tensor, at::Tensor>{grad_weight, grad_bias};
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> sycl_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask)
{
  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    // grad_input = at::sycl_convolution_backward_input(
    //   input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
    // FIXME: we should route to at variable op to insert bp info, even though it is backward op
    // It exists double backward case
    grad_input = sycl_convolution_backward_input(
      input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
  }
  if (output_mask[1] || output_mask[2]) {
    // FIXME: we should route to at variable op to insert bp info, even though it is backward op
    // It exists double backward case
    // std::tie(grad_weight, grad_bias) = at::sycl_convolution_backward_weights(
    //   weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2]);
    std::tie(grad_weight, grad_bias) = sycl_convolution_backward_weights(
      weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2]);
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
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
  bool cudnn_enabled;

  bool is_strided() const;
  bool is_dilated() const;
  bool is_padded() const;
  bool is_output_padding_neg() const;
  bool is_output_padding_big() const;
  bool is_padding_neg() const;
  bool is_stride_nonpos() const;
  void view1d_as_2d();
  bool use_cpu_depthwise3x3_winograd(const at::Tensor& input, const at::Tensor& weight) const;
  bool is_depthwise(const at::Tensor& input, const at::Tensor& weight) const;
};

std::ostream& operator<<(std::ostream & out, const ConvParams& params) {
  out << "ConvParams {"
      << "  stride = " << IntArrayRef{params.stride}
      << "  padding = " << IntArrayRef{params.padding}
      << "  dilation = " << IntArrayRef{params.dilation}
      << "  transposed = " << params.transposed
      << "  output_padding = " << IntArrayRef{params.output_padding}
      << "  groups = " << params.groups
      << "  benchmark = " << params.benchmark
      << "  deterministic = " << params.deterministic
      << "  cudnn_enabled = " << params.cudnn_enabled
      << "}";
  return out;
}

auto ConvParams::is_strided() const -> bool {
  bool is_strided = false;
  for (int s : stride) {
    is_strided |= (s != 1);
  }
  return is_strided;
}

auto ConvParams::is_dilated() const -> bool {
  bool is_dilated = false;
  for (int d : dilation) {
    is_dilated |= (d != 1);
  }
  return is_dilated;
}

auto ConvParams::is_padded() const -> bool {
  bool is_padded = false;
  for (int p : padding) {
    is_padded |= (p != 0);
  }
  return is_padded;
}

auto ConvParams::is_output_padding_neg() const -> bool {
  bool is_non_neg = false;
  for (int p : output_padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}

auto ConvParams::is_output_padding_big() const -> bool {
  bool is_big = false;
  for (size_t i = 0; i < output_padding.size(); i++) {
    is_big |= (output_padding[i] >= stride[i] || output_padding[i] >= dilation[i]);
  }
  return is_big;
}

auto ConvParams::is_padding_neg() const -> bool {
  bool is_non_neg = false;
  for (int p : padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}

auto ConvParams::is_stride_nonpos() const -> bool {
  bool is_nonpos = false;
  for (int s : stride) {
    is_nonpos |= (s <= 0);
  }
  return is_nonpos;
}

auto ConvParams::view1d_as_2d() -> void {
  if (stride.size() == 1) {
    stride.insert(stride.begin(), 1);
    padding.insert(padding.begin(), 0);
    dilation.insert(dilation.begin(), 1);
    output_padding.insert(output_padding.begin(), 0);
  }
}

auto ConvParams::use_cpu_depthwise3x3_winograd(
    const at::Tensor& input, const at::Tensor& weight) const -> bool {
#ifdef __ARM_NEON__
  // Currently only 3x3 depthwise convolutions on tensors of float are supported.
  return (input.ndimension() == 4) &&
         (input.size(1) == groups) &&
         (weight.ndimension() == 4 ) &&
         (weight.size(0) % input.size(1) == 0) &&
         (weight.size(2) == 3) &&
         (weight.size(3) == 3) &&
         (input.device().type() == c10::DeviceType::CPU) &&
         (input.scalar_type() == at::kFloat) &&
         input.is_contiguous() &&
         (weight.device().type() == c10::DeviceType::CPU) &&
         (weight.scalar_type() == at::kFloat) &&
         weight.is_contiguous() &&
         !is_strided() &&
         !is_dilated() &&
         !transposed;
#else
  return false;
#endif
}

auto ConvParams::is_depthwise(
        const at::Tensor& input, const at::Tensor& weight) const -> bool {
  return input.is_cuda() &&
         !transposed &&
         input.ndimension() == 4 &&
         input.size(1) == groups &&
         groups > 1 && // no point if there is only a single group
         weight.size(0) % input.size(1) == 0; // output channels must be a multiple of input channels
}

static void check_shape_forward(const at::Tensor& input,
                                const at::Tensor& weight, const at::Tensor& bias,
                                const ConvParams& params, bool input_is_mkldnn) {
  int64_t k = input.ndimension();
  int64_t weight_dim = weight.ndimension();
  std::vector<int64_t> weight_sizes(weight_dim);
  // mkldnn conv2d weights could have been re-ordered to 5d by
  // mkldnn_reorder_conv2d_weight
  if ((weight_dim == k + 1) && input_is_mkldnn) {
    weight_sizes[0] = weight.size(0) * weight.size(1);
    std::copy_n(
        weight.sizes().cbegin() + 2, k - 1, weight_sizes.begin() + 1);
    weight_dim = k;
  } else {
    std::copy_n(
        weight.sizes().cbegin(), weight_dim, weight_sizes.begin());
  }
  int64_t groups = params.groups;
  auto padding = params.padding;
  auto output_padding = params.output_padding;
  auto stride = params.stride;
  auto dilation = params.dilation;
  bool transposed = params.transposed;

  TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported");
  TORCH_CHECK(!params.is_output_padding_neg(), "negative output_padding is not supported");
  TORCH_CHECK(!params.is_stride_nonpos(), "non-positive stride is not supported");

  TORCH_CHECK(weight_dim == k,
           "Expected ", weight_dim, "-dimensional input for ", weight_dim,
           "-dimensional weight ", weight_sizes, ", but got ", k, "-dimensional input of size ",
           input.sizes(), " instead");
  TORCH_CHECK(weight_sizes[0] >= groups,
           "Given groups=", groups, ", expected weight to be at least ", groups,
           " at dimension 0, but got weight of size ", weight_sizes, " instead");
  TORCH_CHECK(weight_sizes[0] % groups == 0,
           "Given groups=", groups, ", expected weight to be divisible by ",
           groups, " at dimension 0, but got weight of size ", weight_sizes,
           " instead");

  if (!transposed) {
    std::vector<int64_t> input_shape;
    std::vector<int64_t> kernel_shape;
    bool kernel_size_correct = true;

    TORCH_CHECK(input.size(1) == (weight_sizes[1] * groups),
             "Given groups=", groups, ", weight of size ", weight_sizes,
             ", expected input", input.sizes(), " to have ",
             (weight_sizes[1] * groups), " channels, but got ", input.size(1),
             " channels instead");
    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
             "Given weight of size ", weight_sizes,
             ", expected bias to be 1-dimensional with ", weight_sizes[0], " elements",
             ", but got bias of size ", bias.sizes(), " instead");

    for (int i = 2; i < k; ++i) {
      input_shape.push_back(input.size(i) + 2 * padding[i-2]);
      // log new kernel size considering dilation
      kernel_shape.push_back(dilation[i-2] * (weight_sizes[i]-1) + 1);
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }

    TORCH_CHECK(input_shape.size() == kernel_shape.size(), "Inconsistent shape between Input and Kernel");

    if (!kernel_size_correct) {
      // If kernel size is incorrect
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      std::ostringstream output_ss;
      std::string separator = "";

      for (int i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";
      }

      AT_ERROR("Calculated padded input size per channel: (", input_ss.str(), "). "
               "Kernel size: (", kernel_ss.str(), "). Kernel size can't be greater than actual input size");
    }
  } else { // transposed
    TORCH_CHECK(input.size(1) == weight_sizes[0],
             "Given transposed=", transposed, ", weight of size ", weight_sizes,
             ", expected input", input.sizes(), " to have ", weight_sizes[0],
             " channels, but got ", input.size(1), " channels instead");
    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[1] * groups),
             "Given transposed=", transposed, ", weight of size ", weight_sizes,
             ", expected bias to be 1-dimensional with ", weight_sizes[1] * groups, " elements",
             ", but got bias of size ", bias.sizes(), " instead");
  }
}

static auto view4d(const at::Tensor& tensor) -> at::Tensor {
  TORCH_CHECK(tensor.ndimension() == 3,
           "expected 3D tensor, got tensor with ", tensor.ndimension(),
           " dimensions instead");
  return tensor.unsqueeze(2);
}

static auto view3d(const at::Tensor& tensor) -> at::Tensor {
  TORCH_CHECK(tensor.ndimension() == 4,
           "expected 4D tensor, got tensor with ", tensor.ndimension(),
           " dimensions instead");
  return tensor.squeeze(2);
}

} // namespace native
} // namespace at

namespace at {
namespace AtenIpexTypeDPCPP {

using namespace at::native;

Tensor convolution_overrideable(const Tensor & input_r, const Tensor & weight_r, const Tensor & bias_r, IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_, bool transposed_, IntArrayRef output_padding_, int64_t groups_) {
  auto input = input_r.contiguous();
  auto weight = weight_r;
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
  params.output_padding = expand_param_if_needed(output_padding_, "output_padding", dim);
  params.groups = groups_;

  check_shape_forward(input, weight, bias, params, true);

  if (k == 3) {
    params.view1d_as_2d();
    input = view4d(input);
    weight = view4d(weight);
  }

  Tensor output;
  if (params.is_depthwise(input, weight)) {
    auto kernel_size = weight.sizes().slice(2);
    auto stride = params.stride;
    auto padding = params.padding;
    auto dilation = params.dilation;
    output = at::thnn_conv_depthwise2d(input.contiguous(), weight,
        kernel_size, bias, stride, padding, dilation);
  } else {
    output = sycl_convolution(input, weight, bias,
        params.padding, params.stride, params.dilation, params.groups);
  }

  if (k == 3) {
    output = view3d(output);
  }

  return output;
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
