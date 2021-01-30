#include <ATen/ipex_type_dpcpp_customized.h>
#include <ATen/quantized/QTensorImpl.h>
#include <core/DPCPPUtils.h>
//#include <core/Quantizer.h>
#include <core/Runtime.h>
#include <core/TensorImplUtils.h>
#include <tensor/Context.h>
#include <utils/ParamUtils.h>

//#include "Conv.h"
#include "ConvTranspose.h"

using namespace dnnl;
using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

Tensor dpcpp_convolution_transpose(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    int64_t groups) {
  auto input = input_r.contiguous();
  auto weight = weight_r.contiguous();
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});

  auto strm = GpuStreamManager::Instance().get_stream();

  auto ndim = input.ndimension();
  auto output_tz = deconv_output_size(
      input.sizes(), weight.sizes(), padding, stride, dilation, output_padding, groups);
  auto output = at::empty(output_tz, input.options());

  auto src_data_t = dt_to_dnnl(input.scalar_type());
  auto wei_data_t = dt_to_dnnl(weight.scalar_type());
  auto dst_data_t = dt_to_dnnl(output.scalar_type());
  auto bias_data_t = dnnl::memory::data_type::f32;

  if (bias.defined()) {
    bias_data_t = dt_to_dnnl(bias.scalar_type());
  }
  auto usr_bias_data_t = dnnl::memory::data_type::f32;

  auto format_any = memory::format_tag::any;
  auto format_data = deconv_input_fmt(ndim);
  auto format_weight = deconv_weight_fmt(ndim, groups != 1);
  auto format_bias = memory::format_tag::x;

  auto ic = input.size(1);
  auto oc = output_tz[1];
  memory::dims input_tz = input.sizes().vec();
  memory::dims weight_tz =
      deconv_compatible_weight_dims(ndim, groups, oc, ic, weight.sizes());
  memory::dims bias_tz = {oc};
  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = deconv_compatible_dilation(dilation);

  auto input_md = memory::desc(input_tz, src_data_t, format_any);
  auto weight_md = memory::desc(weight_tz, wei_data_t, format_any);
  auto output_md = memory::desc(output_tz, dst_data_t, format_any);
  auto bias_md = bias.defined()
      ? memory::desc(bias_tz, bias_data_t, format_bias)
      : memory::desc();

  auto deconv_forward_desc = deconvolution_forward::desc(
      prop_kind::forward,
      algorithm::deconvolution_direct,
      input_md,
      weight_md,
      bias_md,
      output_md,
      _stride,
      _dilation,
      _padding,
      _padding);

  auto deconv_forward_pd =
      deconvolution_forward::primitive_desc(deconv_forward_desc, engine);

  auto input_usr_memory = dpcpp_onednn_memory(
      {{input_tz}, src_data_t, format_data}, engine, input.data_ptr());

  auto weight_usr_memory = dpcpp_onednn_memory(
      {{weight_tz}, wei_data_t, format_weight}, engine, weight.data_ptr());

  auto output_usr_memory = dpcpp_onednn_memory(
      {{output_tz}, dst_data_t, format_data}, engine, output.data_ptr());

  Tensor input_, weight_, output_;

  auto expected_input_md = deconv_forward_pd.src_desc();
  auto input_memory = input_usr_memory;
  if (input_usr_memory.get_desc() != expected_input_md) {
    auto item_num =
        static_cast<int64_t>(expected_input_md.get_size() / input.itemsize());
    input_ =
        at::AtenIpexTypeXPU::empty({item_num}, input.options(), c10::nullopt);
    input_memory =
        dpcpp_onednn_memory(expected_input_md, engine, input_.data_ptr());
    DPCPP_ONEDNN_EXEC(
        reorder(input_usr_memory, input_memory),
        strm,
        {{DNNL_ARG_FROM, input_usr_memory}, {DNNL_ARG_TO, input_memory}});
  }

  auto expected_weight_md = deconv_forward_pd.weights_desc();
  auto weight_memory = weight_usr_memory;
  if (weight_usr_memory.get_desc() != expected_weight_md) {
    auto item_num =
        static_cast<int64_t>(expected_weight_md.get_size() / weight.itemsize());
    weight_ =
        at::AtenIpexTypeXPU::empty({item_num}, weight.options(), c10::nullopt);
    weight_memory =
        dpcpp_onednn_memory(expected_weight_md, engine, weight_.data_ptr());
    DPCPP_ONEDNN_EXEC(
        reorder(weight_usr_memory, weight_memory),
        strm,
        {{DNNL_ARG_FROM, weight_usr_memory}, {DNNL_ARG_TO, weight_memory}});
  }

  auto expected_output_md = deconv_forward_pd.dst_desc();
  auto output_memory = output_usr_memory;
  if (output_usr_memory.get_desc() != expected_output_md) {
    auto item_num =
        static_cast<int64_t>(expected_output_md.get_size() / output.itemsize());
    output_ =
        at::AtenIpexTypeXPU::empty({item_num}, output.options(), c10::nullopt);
    output_memory =
        dpcpp_onednn_memory(expected_output_md, engine, output_.data_ptr());
  }

  memory bias_memory = memory({{}, bias_data_t, format_bias}, engine);
  if (bias.defined()) {
    bias_memory = dpcpp_onednn_memory(
        {bias_tz, bias_data_t, format_bias}, engine, bias.data_ptr());
  }

  auto deconv_forward = deconvolution_forward(deconv_forward_pd);
  DPCPP_ONEDNN_EXEC(
      deconv_forward,
      strm,
      {{DNNL_ARG_SRC, input_memory},
       {DNNL_ARG_WEIGHTS, weight_memory},
       {DNNL_ARG_BIAS, bias_memory},
       {DNNL_ARG_DST, output_memory}});

  if (output_memory != output_usr_memory) {
    DPCPP_ONEDNN_EXEC(
        reorder(output_memory, output_usr_memory),
        strm,
        {{DNNL_ARG_FROM, output_memory}, {DNNL_ARG_TO, output_usr_memory}});
  }

  return output;
}

Tensor dpcpp_convolution_transpose_backward_input(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& grad_output_r,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined) {
  auto input = input_r.contiguous();
  auto weight = weight_r.contiguous();
  auto grad_output = grad_output_r.contiguous();
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});

  auto strm = GpuStreamManager::Instance().get_stream();

  auto ndim = input.ndimension();
  auto grad_input = at::empty(input.sizes(), grad_output.options());

  auto src_data_t = dt_to_dnnl(input.scalar_type());
  auto wei_data_t = dt_to_dnnl(weight.scalar_type());
  auto dst_data_t = dt_to_dnnl(grad_output.scalar_type());
  auto bias_data_t = dnnl::memory::data_type::f32;
  auto usr_bias_data_t = dnnl::memory::data_type::f32;

  if (bias_defined) {
    bias_data_t = dt_to_dnnl(weight.scalar_type());
  }

  auto format_any = memory::format_tag::any;
  auto format_data = deconv_input_fmt(ndim);
  auto format_weight = deconv_weight_fmt(ndim, groups != 1);
  auto format_bias = memory::format_tag::x;

  auto ic = input.size(1);
  auto oc = grad_output.size(1);
  memory::dims output_tz = grad_output.sizes().vec();
  memory::dims input_tz = input.sizes().vec();
  memory::dims weight_tz =
      deconv_compatible_weight_dims(ndim, groups, oc, ic, weight.sizes());
  memory::dims bias_tz = {oc};
  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = deconv_compatible_dilation(dilation);

  auto input_md = memory::desc(input_tz, src_data_t, format_any);
  auto weight_md = memory::desc(weight_tz, wei_data_t, format_any);
  auto output_md = memory::desc(output_tz, dst_data_t, format_any);
  auto bias_md = bias_defined ? memory::desc(bias_tz, bias_data_t, format_bias)
                              : memory::desc();

  auto deconv_forward_desc = deconvolution_forward::desc(
      prop_kind::forward,
      algorithm::deconvolution_direct,
      input_md,
      weight_md,
      bias_md,
      output_md,
      _stride,
      _dilation,
      _padding,
      _padding);

  auto deconv_forward_pd =
      deconvolution_forward::primitive_desc(deconv_forward_desc, engine);

  auto deconv_backward_data_desc = deconvolution_backward_data::desc(
      algorithm::deconvolution_direct,
      input_md,
      weight_md,
      output_md,
      _stride,
      _dilation,
      _padding,
      _padding);

  auto deconv_backward_data_pd = deconvolution_backward_data::primitive_desc(
      deconv_backward_data_desc, engine, deconv_forward_pd);

  auto grad_output_usr_memory = dpcpp_onednn_memory(
      {{output_tz}, dst_data_t, format_data}, engine, grad_output.data_ptr());

  auto weight_usr_memory = dpcpp_onednn_memory(
      {{weight_tz}, wei_data_t, format_weight}, engine, weight.data_ptr());

  auto grad_input_usr_memory = dpcpp_onednn_memory(
      {{input_tz}, src_data_t, format_data}, engine, grad_input.data_ptr());

  Tensor grad_output_, weight_, grad_input_;
  auto expected_grad_output_md = deconv_backward_data_pd.diff_dst_desc();
  auto grad_output_memory = grad_output_usr_memory;
  if (grad_output_usr_memory.get_desc() != expected_grad_output_md) {
    auto item_num = static_cast<int64_t>(
        expected_grad_output_md.get_size() / grad_output.itemsize());
    grad_output_ = at::AtenIpexTypeXPU::empty(
        {item_num}, grad_output.options(), c10::nullopt);
    grad_output_memory = dpcpp_onednn_memory(
        expected_grad_output_md, engine, grad_output_.data_ptr());
    DPCPP_ONEDNN_EXEC(
        reorder(grad_output_usr_memory, grad_output_memory),
        strm,
        {{DNNL_ARG_FROM, grad_output_usr_memory},
         {DNNL_ARG_TO, grad_output_memory}});
  }

  auto expected_weight_md = deconv_backward_data_pd.weights_desc();
  auto weight_memory = weight_usr_memory;
  if (weight_usr_memory.get_desc() != expected_weight_md) {
    auto item_num =
        static_cast<int64_t>(expected_weight_md.get_size() / weight.itemsize());
    weight_ =
        at::AtenIpexTypeXPU::empty({item_num}, weight.options(), c10::nullopt);
    weight_memory =
        dpcpp_onednn_memory(expected_weight_md, engine, weight_.data_ptr());
    DPCPP_ONEDNN_EXEC(
        reorder(weight_usr_memory, weight_memory),
        strm,
        {{DNNL_ARG_FROM, weight_usr_memory}, {DNNL_ARG_TO, weight_memory}});
  }

  auto expected_grad_input_md = deconv_backward_data_pd.diff_src_desc();
  auto grad_input_memory = grad_input_usr_memory;
  if (grad_input_memory.get_desc() != expected_grad_input_md) {
    auto item_num = static_cast<int64_t>(
        expected_grad_input_md.get_size() / grad_input.itemsize());
    grad_input_ = at::AtenIpexTypeXPU::empty(
        {item_num}, grad_input.options(), c10::nullopt);
    grad_input_memory = dpcpp_onednn_memory(
        expected_grad_input_md, engine, grad_input_.data_ptr());
  }

  auto deconv_backward_data =
      deconvolution_backward_data(deconv_backward_data_pd);
  DPCPP_ONEDNN_EXEC(
      deconv_backward_data,
      strm,
      {{DNNL_ARG_DIFF_DST, grad_output_memory},
       {DNNL_ARG_WEIGHTS, weight_memory},
       {DNNL_ARG_DIFF_SRC, grad_input_memory}});

  if (grad_input_memory != grad_input_usr_memory) {
    DPCPP_ONEDNN_EXEC(
        reorder(grad_input_memory, grad_input_usr_memory),
        strm,
        {{DNNL_ARG_FROM, grad_input_memory},
         {DNNL_ARG_TO, grad_input_usr_memory}});
  }
  return grad_input;
}

std::tuple<at::Tensor, at::Tensor> dpcpp_convolution_transpose_backward_weights(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& grad_output_r,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined) {
  auto input = input_r.contiguous();
  auto weight = weight_r.contiguous();
  auto grad_output = grad_output_r.contiguous();
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto ndim = input.ndimension();
  auto grad_weight = at::empty(weight.sizes(), grad_output.options());

  auto src_data_t = dt_to_dnnl(input.scalar_type());
  auto wei_data_t = dt_to_dnnl(weight.scalar_type());
  auto dst_data_t = dt_to_dnnl(grad_output.scalar_type());
  auto bias_data_t = dnnl::memory::data_type::f32;
  auto usr_bias_data_t = dnnl::memory::data_type::f32;

  Tensor grad_bias;
  if (bias_defined) {
    grad_bias = at::empty({grad_output.size(1)}, grad_output.options());
    bias_data_t = dt_to_dnnl(weight.scalar_type());
  }

  auto format_any = memory::format_tag::any;
  auto format_data = deconv_input_fmt(ndim);
  auto format_weight = deconv_weight_fmt(ndim, groups != 1);
  auto format_bias = memory::format_tag::x;

  auto ic = input.size(1);
  auto oc = grad_output.size(1);
  memory::dims output_tz = grad_output.sizes().vec();
  memory::dims input_tz = input.sizes().vec();
  memory::dims weight_tz =
      deconv_compatible_weight_dims(ndim, groups, oc, ic, weight.sizes());
  memory::dims bias_tz = {oc};
  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = deconv_compatible_dilation(dilation);

  auto input_md = memory::desc(input_tz, src_data_t, format_any);
  auto weight_md = memory::desc(weight_tz, wei_data_t, format_any);
  auto output_md = memory::desc(output_tz, dst_data_t, format_any);
  auto bias_md = bias_defined ? memory::desc(bias_tz, bias_data_t, format_bias)
                              : memory::desc();

  auto deconv_forward_desc = deconvolution_forward::desc(
      prop_kind::forward,
      algorithm::deconvolution_direct,
      input_md,
      weight_md,
      bias_md,
      output_md,
      _stride,
      _dilation,
      _padding,
      _padding);

  auto deconv_forward_pd =
      deconvolution_forward::primitive_desc(deconv_forward_desc, engine);

  auto deconv_backward_weight_desc = deconvolution_backward_weights::desc(
      algorithm::deconvolution_direct,
      input_md,
      weight_md,
      bias_md,
      output_md,
      _stride,
      _dilation,
      _padding,
      _padding);

  auto deconv_backward_weights_pd =
      deconvolution_backward_weights::primitive_desc(
          deconv_backward_weight_desc, engine, deconv_forward_pd);

  auto input_usr_memory = dpcpp_onednn_memory(
      {{input_tz}, src_data_t, format_data}, engine, input.data_ptr());

  auto grad_output_usr_memory = dpcpp_onednn_memory(
      {{output_tz}, dst_data_t, format_data}, engine, grad_output.data_ptr());

  auto grad_weight_usr_memory = dpcpp_onednn_memory(
      {{weight_tz}, wei_data_t, format_weight}, engine, grad_weight.data_ptr());

  auto grad_bias_memory = bias_defined
      ? dpcpp_onednn_memory(
            {{bias_tz}, bias_data_t, format_bias}, engine, grad_bias.data_ptr())
      : memory({{}, bias_data_t, format_bias}, engine);

  Tensor input_, grad_output_, grad_weight_, grad_bias_;
  auto expected_input_md = deconv_backward_weights_pd.src_desc();
  auto input_memory = input_usr_memory;
  if (input_usr_memory.get_desc() != expected_input_md) {
    auto item_num =
        static_cast<int64_t>(expected_input_md.get_size() / input.itemsize());
    input_ =
        at::AtenIpexTypeXPU::empty({item_num}, input.options(), c10::nullopt);
    input_memory =
        dpcpp_onednn_memory(expected_input_md, engine, input_.data_ptr());
    DPCPP_ONEDNN_EXEC(
        reorder(input_usr_memory, input_memory),
        strm,
        {{DNNL_ARG_FROM, input_usr_memory}, {DNNL_ARG_TO, input_memory}});
  }

  auto expected_grad_output_md = deconv_backward_weights_pd.diff_dst_desc();
  auto grad_output_memory = grad_output_usr_memory;
  if (grad_output_usr_memory.get_desc() != expected_grad_output_md) {
    auto item_num = static_cast<int64_t>(
        expected_grad_output_md.get_size() / grad_output.itemsize());
    grad_output_ = at::AtenIpexTypeXPU::empty(
        {item_num}, grad_output.options(), c10::nullopt);
    grad_output_memory = dpcpp_onednn_memory(
        expected_grad_output_md, engine, grad_output_.data_ptr());
    DPCPP_ONEDNN_EXEC(
        reorder(grad_output_usr_memory, grad_output_memory),
        strm,
        {{DNNL_ARG_FROM, grad_output_usr_memory},
         {DNNL_ARG_TO, grad_output_memory}});
  }

  auto expected_grad_weight_md = deconv_backward_weights_pd.diff_weights_desc();
  auto grad_weight_memory = grad_weight_usr_memory;
  if (grad_weight_usr_memory.get_desc() != expected_grad_weight_md) {
    auto item_num = static_cast<int64_t>(
        expected_grad_weight_md.get_size() / grad_weight.itemsize());
    grad_weight_ = at::AtenIpexTypeXPU::empty(
        {item_num}, grad_weight.options(), c10::nullopt);
    grad_weight_memory = dpcpp_onednn_memory(
        expected_grad_weight_md, engine, grad_weight_.data_ptr());
  }

  auto deconv_backward_weights =
      deconvolution_backward_weights(deconv_backward_weights_pd);
  DPCPP_ONEDNN_EXEC(
      deconv_backward_weights,
      strm,
      {{DNNL_ARG_DIFF_DST, grad_output_memory},
       {DNNL_ARG_SRC, input_memory},
       {DNNL_ARG_DIFF_WEIGHTS, grad_weight_memory},
       {DNNL_ARG_DIFF_BIAS, grad_bias_memory}});

  if (grad_weight_memory != grad_weight_usr_memory) {
    DPCPP_ONEDNN_EXEC(
        reorder(grad_weight_memory, grad_weight_usr_memory),
        strm,
        {{DNNL_ARG_FROM, grad_weight_memory},
         {DNNL_ARG_TO, grad_weight_usr_memory}});
  }
  return std::tuple<at::Tensor, at::Tensor>{grad_weight, grad_bias};
}

} // namespace impl
} // namespace AtenIpexTypeXPU
} // namespace at
