#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#include <c10/dpcpp/SYCLUtils.h>

#if !AT_MKLDNN_ENABLED() && !AT_SYCL_ENABLED()

namespace at { namespace native {

at::Tensor mkldnn_convolution_forward(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    IntList padding, IntList stride, IntList dilation, int64_t groups) {
  AT_ERROR("mkldnn_convolution_forward: ATen not compiled with MKLDNN support");
}

at::Tensor mkldnn_convolution_backward_input(
    IntList input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntList padding, IntList stride, IntList dilation, int64_t groups, bool bias_defined) {
  AT_ERROR("mkldnn_convolution_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<at::Tensor,at::Tensor> mkldnn_convolution_backward_weights(
    IntList weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntList padding, IntList stride, IntList dilation, int64_t groups, bool bias_defined) {
  AT_ERROR("mkldnn_convolution_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> mkldnn_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntList padding, IntList stride, IntList dilation, int64_t groups, std::array<bool,3> output_mask) {
  AT_ERROR("mkldnn_convolution_backward: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_EBABLED || AT_SYCL_ENABLED

#include <ATen/dpcpp/Runtime.h>

using namespace mkldnn;

namespace at { namespace native {

constexpr int input_batch_size_dim = 0;  // also grad_input
// constexpr int input_channels_dim = 1;
// constexpr int output_batch_size_dim = 0;  // also grad_output
// constexpr int output_channels_dim = 1;
constexpr int weight_output_channels_dim = 0;
// constexpr int weight_input_channels_dim = 1;

// Often written as 2 + max_dim (extra dims for batch size and channels)
// constexpr int max_dim = 3;

static std::vector<int64_t> conv_output_size(
    IntList input_size, IntList weight_size,
    IntList padding, IntList stride, IntList dilation, int64_t groups)
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

// XXX: The name of mkldnn_convolution is refined in mkldnn/include/mkldnn_dnnl_mangling.h.
// We have to rename it here to avoid linking error.
at::Tensor mkldnn_convolution_forward(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    IntList padding, IntList stride, IntList dilation, int64_t groups)
{
  auto output = at::empty(conv_output_size(
    input.sizes(), weight.sizes(), padding, stride, dilation, groups), input.options());

#if AT_SYCL_ENABLED()
  Device curDevice = Device(kSYCL, c10::sycl::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
#else
  auto engine = CpuEngine::Instance().get_engine();
#endif

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

#if AT_SYCL_ENABLED()
  auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(input.data_ptr(), input_usr_memory);

  auto weight_usr_memory = memory({{{weight_tz}, data_t,  format_weight}, engine});
  sycl_set_mkldnn_buffer(weight.data_ptr(), weight_usr_memory);

  auto output_usr_memory = memory({{{output_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(output.data_ptr(), output_usr_memory);

#else
  auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine},
    input.data_ptr());
  auto weight_usr_memory = memory({{{weight_tz}, data_t,  format_weight}, engine},
    weight.data_ptr());
  auto output_usr_memory = memory({{{output_tz}, data_t, format_nchw}, engine},
    output.data_ptr());
#endif

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
#if AT_SYCL_ENABLED()
    bias_usr_memory.reset(new memory({{{bias_tz}, data_t, format_x}, engine}));
    sycl_set_mkldnn_buffer(bias.data_ptr(), *bias_usr_memory);
#else
    bias_usr_memory.reset(new memory({{{bias_tz}, data_t, format_x}, engine},
      bias.data_ptr()));
#endif
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

at::Tensor mkldnn_convolution_backward_input(
    IntList input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntList padding, IntList stride, IntList dilation, int64_t groups, bool bias_defined)
{
  auto grad_input = at::empty(input_size, grad_output.options());

#if AT_SYCL_ENABLED()
  Device curDevice = Device(kSYCL, c10::sycl::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
#else
  auto engine = CpuEngine::Instance().get_engine();
#endif

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

#if AT_SYCL_ENABLED()
  auto grad_output_usr_memory = memory({{{output_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(grad_output.data_ptr(), grad_output_usr_memory);

  auto weight_usr_memory = memory({{{weight_tz}, data_t, format_weight}, engine});
  sycl_set_mkldnn_buffer(weight.data_ptr(), weight_usr_memory);

  auto grad_input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(grad_input.data_ptr(), grad_input_usr_memory);
#else
  auto grad_output_usr_memory = memory({{{output_tz}, data_t, format_nchw}, engine},
    grad_output.data_ptr());
  auto weight_usr_memory = memory({{{weight_tz}, data_t, format_weight}, engine},
    weight.data_ptr());
  auto grad_input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine},
    grad_input.data_ptr());
#endif

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

std::tuple<at::Tensor, at::Tensor> mkldnn_convolution_backward_weights(
    IntList weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntList padding, IntList stride, IntList dilation, int64_t groups, bool bias_defined)
{
  auto grad_weight = at::empty(weight_size, grad_output.options());

  Tensor grad_bias;
  if (bias_defined) {
    grad_bias = at::empty({grad_output.size(1)}, grad_output.options());
  }

#if AT_SYCL_ENABLED()
  Device curDevice = Device(kSYCL, c10::sycl::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
#else
  auto engine = CpuEngine::Instance().get_engine();
#endif

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

#if AT_SYCL_ENABLED()
  auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(input.data_ptr(), input_usr_memory);

  auto grad_output_usr_memory = memory({{{output_tz}, data_t, format_nchw}, engine});
  sycl_set_mkldnn_buffer(grad_output.data_ptr(), grad_output_usr_memory);

  auto grad_weight_usr_memory = memory({{{weight_tz}, data_t, format_weight}, engine});
  sycl_set_mkldnn_buffer(grad_weight.data_ptr(), grad_weight_usr_memory);
#else
  auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, engine},
    input.data_ptr());
  auto grad_output_usr_memory = memory({{{output_tz}, data_t, format_nchw}, engine},
    grad_output.data_ptr());
  auto grad_weight_usr_memory = memory({{{weight_tz}, data_t, format_weight}, engine},
    grad_weight.data_ptr());
#endif

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
#if AT_SYCL_ENABLED()
    grad_bias_memory.reset(new memory({{{bias_tz}, data_t, format_x}, engine}));
    sycl_set_mkldnn_buffer(grad_bias.data_ptr(), *grad_bias_memory);
#else
    grad_bias_memory.reset(new memory({{{bias_tz}, data_t, format_x}, engine},
      grad_bias.data_ptr()));
#endif
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

std::tuple<at::Tensor,at::Tensor,at::Tensor> mkldnn_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntList padding, IntList stride, IntList dilation, int64_t groups, std::array<bool,3> output_mask)
{
  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::mkldnn_convolution_backward_input(
      input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::mkldnn_convolution_backward_weights(
      weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2]);
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

}}  // namespace at::native

#endif
