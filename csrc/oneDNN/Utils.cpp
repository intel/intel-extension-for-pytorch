#include <oneapi/dnnl/dnnl.hpp>
#include <tensor/Context.h>
#include "oneDNN.h"

namespace xpu {
namespace oneDNN {

// used for weight prepack Linear in torch.xpu.optimize
at::Tensor convert_linear_weight_layout(
    at::Tensor& weight,
    const IntArrayRef input_size) {
  auto output_feature = weight.size(0);
  auto input_feature = weight.size(1);

  // if input size is [], use dummy batch size
  auto batch_size = input_size.size() ? input_size[0] : 256;

  weight = weight.t().contiguous();

  auto weight_dt = get_onednn_dtype(weight);
  auto input_dt = weight_dt;
  auto output_dt = weight_dt;

  auto m1_any_md = memory::desc(
      {batch_size, input_feature}, input_dt, memory::format_tag::any);
  auto m2_any_md = memory::desc(
      {input_feature, output_feature}, weight_dt, memory::format_tag::any);
  auto dst_any_md = memory::desc(
      {batch_size, output_feature}, output_dt, memory::format_tag::any);

  auto matmul_desc = matmul::desc(m1_any_md, m2_any_md, dst_any_md);

  primitive_attr pattr;
#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

  auto matmul_pd = matmul::primitive_desc(
      matmul_desc,
      pattr,
      GpuEngineManager::Instance().get_engine(
          at::Device(at::kXPU, current_device())));

  memory::desc plain_md = get_onednn_md(weight);
  memory::desc expected_md = matmul_pd.weights_desc();

  Tensor weight_;
  if (plain_md != expected_md) {
    auto weight_opt = weight.options();
    weight_ = empty_opaque_tensor(expected_md, weight_opt, c10::nullopt);
    xpu::oneDNN::reorder(weight, weight_);
    auto weight_opt_ctx = DPCPPTensorContext::release_tensor_ctx(weight_);
    DPCPPTensorContext::set_tensor_ctx(weight, std::move(weight_opt_ctx));
  } else {
    // if no need to reorder, the weight needs to be recovered
    weight = weight.t().contiguous();
  }

  return weight;
}

// used for weight prepack Conv in torch.xpu.optimize
void convert_conv_weight_layout(
    const at::Tensor& weight,
    const IntArrayRef padding,
    const IntArrayRef stride,
    IntArrayRef dilation,
    const int64_t groups,
    const IntArrayRef input_size) {
  auto weight_ndim = weight.ndimension();
  TORCH_CHECK(
      3 == weight_ndim || 4 == weight_ndim || 5 == weight_ndim,
      "convolution only uses 3D, 4D and 5D weight");

  // if input size is [], use dummy input
  auto input_size_for_prepack = input_size.size()
      ? input_size.vec()
      : gen_dummy_input_size_for(weight.sizes(), groups);

  at::Tensor input;
  if (3 == weight_ndim) {
    // Conv1d
    input = at::empty({input_size_for_prepack}, weight.options());
  } else if (weight_ndim == 4) {
    // Conv2d
    input = at::empty({input_size_for_prepack}, weight.options());
  } else {
    // Conv3d
    input = at::empty({input_size_for_prepack}, weight.options());
  }

  // bias shape is oc, suppose weight and bias has same datatype
  auto bias = at::empty({weight.sizes()[0]}, weight.options());

  auto conv_pd = get_convolution_pd(
      input, weight, bias, padding, stride, dilation, groups);

  memory::desc plain_md = get_onednn_md(weight);
  memory::desc expected_md = conv_pd.weights_desc();

  Tensor weight_;
  if (plain_md != expected_md) {
    auto weight_opt = weight.options();
    weight_ = empty_opaque_tensor(expected_md, weight_opt, c10::nullopt);
    xpu::oneDNN::reorder(weight, weight_);
    auto weight_opt_ctx = DPCPPTensorContext::release_tensor_ctx(weight_);
    DPCPPTensorContext::set_tensor_ctx(weight, std::move(weight_opt_ctx));
  }
}

// used for weight prepack DeConv in torch.xpu.optimize
void convert_convtranspose_weight_layout(
    const at::Tensor& weight,
    const IntArrayRef padding,
    const IntArrayRef stride,
    IntArrayRef dilation,
    const IntArrayRef dst_padding,
    const int64_t groups,
    const IntArrayRef input_size) {
  auto weight_ndim = weight.ndimension();
  TORCH_CHECK(
      3 == weight_ndim || 4 == weight_ndim || 5 == weight_ndim,
      "convolution transpose only uses 3D, 4D and 5D weight");

  // if input size is [], use dummy input
  auto input_size_for_prepack = input_size.size()
      ? input_size.vec()
      : gen_dummy_input_size_for(weight.sizes(), groups);

  at::Tensor input;
  if (weight_ndim == 3) {
    // DeConv1d
    input = at::empty({input_size_for_prepack}, weight.options());
  } else if (weight_ndim == 4) {
    // DeConv2d
    input = at::empty({input_size_for_prepack}, weight.options());
  } else {
    // DeConv3d
    input = at::empty({input_size_for_prepack}, weight.options());
  }

  auto ndim = input.ndimension();
  auto dst_tz = deconv_dst_size(
      input.sizes(),
      weight.sizes(),
      padding,
      stride,
      dilation,
      dst_padding,
      groups);

  auto ic = input.size(1);
  auto oc = dst_tz[1];

  memory::dims input_tz = input.sizes().vec();
  memory::dims weight_tz =
      deconv_compatible_wgh_dims(ndim, groups, oc, ic, weight.sizes());
  memory::dims bias_tz = {oc};

  auto format_any = memory::format_tag::any;

  memory::data_type input_dt = get_onednn_dtype(input);
  memory::data_type weight_dt = get_onednn_dtype(weight);
  memory::data_type dst_dt = input_dt;
  // suppose weight and bias has same datatype
  memory::data_type bias_dt = weight_dt;

  auto input_md = memory::desc(input_tz, input_dt, format_any);
  auto weight_md = memory::desc(weight_tz, weight_dt, format_any);
  auto dst_md = memory::desc(dst_tz, dst_dt, format_any);
  auto bias_md = memory::desc(bias_tz, bias_dt, format_any);

  auto deconv_fwd_desc = deconvolution_forward::desc(
      prop_kind::forward,
      algorithm::deconvolution_direct,
      input_md,
      weight_md,
      bias_md,
      dst_md,
      stride.vec(),
      deconv_compatible_dilation(dilation),
      padding.vec(),
      padding.vec());

  auto deconv_pd = deconvolution_forward::primitive_desc(
      deconv_fwd_desc,
      GpuEngineManager::Instance().get_engine({kXPU, current_device()}));

  memory::desc plain_md = get_onednn_md(weight);
  memory::desc expected_md = deconv_pd.weights_desc();

  Tensor weight_;
  if (plain_md != expected_md) {
    auto weight_opt = weight.options();
    weight_ = empty_opaque_tensor(expected_md, weight_opt, c10::nullopt);
    xpu::oneDNN::reorder(weight, weight_);
    auto weight_opt_ctx = DPCPPTensorContext::release_tensor_ctx(weight_);
    DPCPPTensorContext::set_tensor_ctx(weight, std::move(weight_opt_ctx));
  }
}

} // namespace oneDNN
} // namespace xpu