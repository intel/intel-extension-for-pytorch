#include <intrinsic/ipex_intrinsic.h>
#include <ATen/quantized/QTensorImpl.h>
#include <runtime/Utils.h>
#include <core/TensorImplUtils.h>
#include <tensor/Context.h>
#include "comm/ParamUtils.h"
#include <oneDNN/oneDNN.h>

#include "ConvTranspose.h"


using namespace dnnl;
using namespace at::native;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

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

  auto src_data_t = get_onednn_dtype(input);
  auto wei_data_t = get_onednn_dtype(weight);
  auto dst_data_t = get_onednn_dtype(output);
  auto bias_data_t = dnnl::memory::data_type::f32;

  if (bias.defined()) {
    bias_data_t = get_onednn_dtype(bias);
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

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key_pd;
  create_key(key_pd, input_md, weight_md, bias.defined(), dst_data_t,
      _stride, _dilation, _padding, _padding);
#endif

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

  memory weight_usr_memory;
  memory::desc input_usr_md, weight_usr_md, output_usr_md;
  if (!Settings::I().is_onednn_layout_enabled()) {
    input_usr_md = memory::desc({input_tz}, src_data_t, format_data);
    weight_usr_md = memory::desc({weight_tz}, wei_data_t, format_weight);
    output_usr_md = memory::desc({output_tz}, dst_data_t, format_data);
  } else {
    auto input_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(input);
    input_usr_md = input_ctx.is_plain()
        ? memory::desc({input_tz}, src_data_t, format_data)
        : input_ctx.meta();

    auto weight_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(weight);
    weight_usr_md = weight_ctx.is_plain()
        ? memory::desc({weight_tz}, wei_data_t, format_weight)
        : weight_ctx.meta();

    auto output_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(output);
    output_usr_md = output_ctx.is_plain()
        ? memory::desc({output_tz}, dst_data_t, format_data)
        : output_ctx.meta();
  }

  Tensor input_, weight_, output_ = output;
  auto expected_input_md = deconv_forward_pd.src_desc();
  auto input_memory = dpcpp_onednn_memory(input_usr_md, engine, input.data_ptr());
  if (input_usr_md != expected_input_md) {
    input_ = empty_opaque_tensor(expected_input_md, input.options(), c10::nullopt);
    input_memory = dpcpp_onednn_memory(expected_input_md, engine, input_.data_ptr());
    xpu::oneDNN::reorder(input, input_);
  }

  auto expected_weight_md = deconv_forward_pd.weights_desc();
  weight_usr_memory = dpcpp_onednn_memory(weight_usr_md, engine, weight.data_ptr());
  auto weight_memory = weight_usr_memory;
  if (weight_usr_memory.get_desc() != expected_weight_md) {
    auto item_num =
        static_cast<int64_t>(expected_weight_md.get_size() / weight.itemsize());
    weight_ =
        at::AtenIpexTypeXPU::empty({item_num}, weight.options(), c10::nullopt);
    weight_memory =
        dpcpp_onednn_memory(expected_weight_md, engine, weight_.data_ptr());

   // Workaround for weight format: iodhw, which onednn can't
   // generate desc correctly by using stride.
   // Track JIRA: https://jira.devtools.intel.com/browse/MFDNN-4958
#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, weight_usr_md, expected_weight_md);
  auto reorder_p = fetch_or_create_m<dnnl::reorder>(key, weight_usr_memory, weight_memory);
#else
  auto reorder_p = dnnl::reorder(weight_usr_memory, weight_memory);
#endif
    DPCPP_ONEDNN_EXEC(
        reorder_p,
        strm,
        {{DNNL_ARG_FROM, weight_usr_memory}, {DNNL_ARG_TO, weight_memory}});
  }

  auto expected_output_md = deconv_forward_pd.dst_desc();
  auto output_memory = dpcpp_onednn_memory(output_usr_md, engine, output.data_ptr());
  if (output_usr_md != expected_output_md) {
    output_ = empty_opaque_tensor(expected_output_md, output.options(), c10::nullopt);
    output_memory =
        dpcpp_onednn_memory(expected_output_md, engine, output_.data_ptr());
  }

  memory bias_memory = memory({{}, bias_data_t, format_bias}, engine);
  if (bias.defined()) {
    bias_memory = dpcpp_onednn_memory(
        {bias_tz, bias_data_t, format_bias}, engine, bias.data_ptr());
  }

#ifdef USE_PRIMITIVE_CACHE
  auto deconv_forward = fetch_or_create_m<deconvolution_forward>(key_pd, deconv_forward_pd);
#else
  auto deconv_forward = deconvolution_forward(deconv_forward_pd);
#endif
  DPCPP_ONEDNN_EXEC(
      deconv_forward,
      strm,
      {{DNNL_ARG_SRC, input_memory},
       {DNNL_ARG_WEIGHTS, weight_memory},
       {DNNL_ARG_BIAS, bias_memory},
       {DNNL_ARG_DST, output_memory}});

  if (!Settings::I().is_onednn_layout_enabled() && output_.data_ptr() != output.data_ptr()) {
    xpu::oneDNN::reorder(output_, output);
  } else if (Settings::I().is_onednn_layout_enabled() && output_.data_ptr() != output.data_ptr()) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(output_);
    DPCPPTensorContext::set_tensor_ctx(output, std::move(blk_ctx));
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

  auto src_data_t = get_onednn_dtype(input);
  auto wei_data_t = get_onednn_dtype(weight);
  auto dst_data_t = get_onednn_dtype(grad_output);
  auto bias_data_t = dnnl::memory::data_type::f32;
  auto usr_bias_data_t = dnnl::memory::data_type::f32;

  if (bias_defined) {
    bias_data_t = get_onednn_dtype(weight);
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

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key_pd;
  create_key(key_pd, input_md, weight_md, bias_defined, dst_data_t,
      _stride, _dilation, _padding, _padding);
#endif

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

  memory weight_usr_memory;
  memory::desc grad_output_usr_md, weight_usr_md, grad_input_usr_md;
  if (!Settings::I().is_onednn_layout_enabled()) {
    grad_output_usr_md = memory::desc({output_tz}, dst_data_t, format_data);
    weight_usr_md = memory::desc({weight_tz}, wei_data_t, format_weight);
    grad_input_usr_md = memory::desc({input_tz}, src_data_t, format_data);
  } else {
    auto grad_output_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(grad_output);
    grad_output_usr_md = grad_output_ctx.is_plain()
        ? memory::desc({output_tz}, dst_data_t, format_data)
        : grad_output_ctx.meta();

    auto weight_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(weight);
    weight_usr_md = weight_ctx.is_plain()
        ? memory::desc({weight_tz}, wei_data_t, format_weight)
        : weight_ctx.meta();

    auto grad_input_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(grad_input);
    grad_input_usr_md = grad_input_ctx.is_plain()
        ? memory::desc({input_tz}, src_data_t, format_data)
        : grad_input_ctx.meta();
  }

  Tensor grad_output_, weight_, grad_input_ = grad_input;
  auto expected_grad_output_md = deconv_backward_data_pd.diff_dst_desc();
  auto grad_output_memory = dpcpp_onednn_memory(grad_output_usr_md, engine, grad_output.data_ptr());
  if (grad_output_usr_md != expected_grad_output_md) {
    grad_output_ = empty_opaque_tensor(expected_grad_output_md, grad_output.options(), c10::nullopt);
    grad_output_memory = dpcpp_onednn_memory(expected_grad_output_md, engine, grad_output_.data_ptr());
    xpu::oneDNN::reorder(grad_output, grad_output_);
  }

  auto expected_weight_md = deconv_backward_data_pd.weights_desc();
  weight_usr_memory = dpcpp_onednn_memory(weight_usr_md, engine, weight.data_ptr());
  auto weight_memory = weight_usr_memory;
  if (weight_usr_memory.get_desc() != expected_weight_md) {
    auto item_num =
        static_cast<int64_t>(expected_weight_md.get_size() / weight.itemsize());
    weight_ =
        at::AtenIpexTypeXPU::empty({item_num}, weight.options(), c10::nullopt);
    weight_memory =
        dpcpp_onednn_memory(expected_weight_md, engine, weight_.data_ptr());

#ifdef USE_PRIMITIVE_CACHE
    lru_key_t key;
    create_key(key, weight_usr_md, expected_weight_md);
    auto reorder_p = fetch_or_create_m<dnnl::reorder>(key, weight_usr_memory, weight_memory);
#else
    auto reorder_p = dnnl::reorder(weight_usr_memory, weight_memory);
#endif
    DPCPP_ONEDNN_EXEC(
        reorder_p,
        strm,
        {{DNNL_ARG_FROM, weight_usr_memory}, {DNNL_ARG_TO, weight_memory}});
  }

  auto expected_grad_input_md = deconv_backward_data_pd.diff_src_desc();
  auto grad_input_memory = dpcpp_onednn_memory(grad_input_usr_md, engine, grad_input.data_ptr());
  if (grad_input_usr_md != expected_grad_input_md) {
    grad_input_ = empty_opaque_tensor(expected_grad_input_md, grad_input.options(), c10::nullopt);
    grad_input_memory = dpcpp_onednn_memory(expected_grad_input_md, engine, grad_input_.data_ptr());
  }

#ifdef USE_PRIMITIVE_CACHE
  auto deconv_backward_data = fetch_or_create_m<deconvolution_backward_data>(
      key_pd, deconv_backward_data_pd);
#else
  auto deconv_backward_data =
      deconvolution_backward_data(deconv_backward_data_pd);
#endif
  DPCPP_ONEDNN_EXEC(
      deconv_backward_data,
      strm,
      {{DNNL_ARG_DIFF_DST, grad_output_memory},
       {DNNL_ARG_WEIGHTS, weight_memory},
       {DNNL_ARG_DIFF_SRC, grad_input_memory}});

  if (!Settings::I().is_onednn_layout_enabled() && grad_input_.data_ptr() != grad_input.data_ptr()) {
    xpu::oneDNN::reorder(grad_input_, grad_input);
  } else if(Settings::I().is_onednn_layout_enabled() && grad_input_.data_ptr() != grad_input.data_ptr()) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(grad_input_);
    DPCPPTensorContext::set_tensor_ctx(grad_input, std::move(blk_ctx));
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

  auto src_data_t = get_onednn_dtype(input);
  auto wei_data_t = get_onednn_dtype(weight);
  auto dst_data_t = get_onednn_dtype(grad_output);
  auto bias_data_t = dnnl::memory::data_type::f32;
  auto usr_bias_data_t = dnnl::memory::data_type::f32;

  Tensor grad_bias;
  if (bias_defined) {
    grad_bias = at::empty({grad_output.size(1)}, grad_output.options());
    bias_data_t = get_onednn_dtype(weight);
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

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key_pd;
  create_key(key_pd, input_md, weight_md, bias_defined, dst_data_t,
      _stride, _dilation, _padding, _dilation);
#endif

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

  memory grad_weight_usr_memory, grad_bias_memory;
  memory::desc input_usr_md, grad_output_usr_md, grad_weight_usr_md, grad_bias_md;
  if (!Settings::I().is_onednn_layout_enabled()) {
    input_usr_md = memory::desc({input_tz}, src_data_t, format_data);
    grad_output_usr_md = memory::desc({output_tz}, dst_data_t, format_data);
    grad_weight_usr_md = memory::desc({weight_tz}, wei_data_t, format_weight);
    grad_bias_md = bias_defined
        ? memory::desc({bias_tz}, bias_data_t, format_bias)
        : memory::desc({}, bias_data_t, format_bias);
  } else {
    auto input_usr_ctx =
        at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(input);
    input_usr_md = input_usr_ctx.is_plain()
        ? memory::desc({input_tz}, src_data_t, format_data)
        : input_usr_ctx.meta();

    auto grad_output_usr_ctx =
        at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(grad_output);
    grad_output_usr_md = grad_output_usr_ctx.is_plain()
        ? memory::desc({output_tz}, dst_data_t, format_data)
        : grad_output_usr_ctx.meta();

    auto grad_weight_usr_ctx =
        at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(grad_weight);
    grad_weight_usr_md = grad_weight_usr_ctx.is_plain()
        ? memory::desc({weight_tz}, wei_data_t, format_weight)
        : grad_weight_usr_ctx.meta();

    if (bias_defined) {
        auto grad_bias_ctx =
            at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(grad_bias);
        grad_bias_md = grad_bias_ctx.is_plain()
            ? memory::desc({bias_tz}, bias_data_t, format_bias)
            : grad_bias_ctx.meta();
    }
  }

  if (bias_defined){
    grad_bias_memory = dpcpp_onednn_memory(grad_bias_md, engine, grad_bias.data_ptr());
  } else {
    grad_bias_memory = memory({{}, bias_data_t, format_bias}, engine);
  }

  Tensor input_, grad_output_, grad_weight_ = grad_weight, grad_bias_;
  auto expected_input_md = deconv_backward_weights_pd.src_desc();
  auto input_memory = dpcpp_onednn_memory(input_usr_md, engine, input.data_ptr());
  if (input_usr_md != expected_input_md) {
    input_ = empty_opaque_tensor(expected_input_md, input.options(), c10::nullopt);
    input_memory = dpcpp_onednn_memory(expected_input_md, engine, input_.data_ptr());
    xpu::oneDNN::reorder(input, input_);
  }

  auto expected_grad_output_md = deconv_backward_weights_pd.diff_dst_desc();
  auto grad_output_memory = dpcpp_onednn_memory(grad_output_usr_md, engine, grad_output.data_ptr());
  if (grad_output_usr_md != expected_grad_output_md) {
    grad_output_ = empty_opaque_tensor(expected_grad_output_md, grad_output.options(), c10::nullopt);
    grad_output_memory = dpcpp_onednn_memory(expected_grad_output_md, engine, grad_output_.data_ptr());
    xpu::oneDNN::reorder(grad_output, grad_output_);
  }

  auto expected_grad_weight_md = deconv_backward_weights_pd.diff_weights_desc();
  grad_weight_usr_memory = dpcpp_onednn_memory(grad_weight_usr_md, engine, grad_weight.data_ptr());
  auto grad_weight_memory = grad_weight_usr_memory;
  if (grad_weight_usr_memory.get_desc() != expected_grad_weight_md) {
    auto item_num = static_cast<int64_t>(
        expected_grad_weight_md.get_size() / grad_weight.itemsize());
    grad_weight_ = at::AtenIpexTypeXPU::empty(
        {item_num}, grad_weight.options(), c10::nullopt);
    grad_weight_memory = dpcpp_onednn_memory(
        expected_grad_weight_md, engine, grad_weight_.data_ptr());
  }

#ifdef USE_PRIMITIVE_CACHE
  auto deconv_backward_weights = fetch_or_create_m<deconvolution_backward_weights>(
      key_pd, deconv_backward_weights_pd);
#else
  auto deconv_backward_weights =
      deconvolution_backward_weights(deconv_backward_weights_pd);
#endif

  DPCPP_ONEDNN_EXEC(
      deconv_backward_weights,
      strm,
      {{DNNL_ARG_DIFF_DST, grad_output_memory},
       {DNNL_ARG_SRC, input_memory},
       {DNNL_ARG_DIFF_WEIGHTS, grad_weight_memory},
       {DNNL_ARG_DIFF_BIAS, grad_bias_memory}});

  if (grad_weight_.data_ptr() != grad_weight.data_ptr()) {
#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, expected_grad_weight_md, grad_weight_usr_md);
  auto reorder_p = fetch_or_create_m<dnnl::reorder>(key, grad_weight_memory, grad_weight_usr_memory);
#else
  auto reorder_p = dnnl::reorder(grad_weight_memory, grad_weight_usr_memory);
#endif
    DPCPP_ONEDNN_EXEC(
        reorder_p,
        strm,
        {{DNNL_ARG_FROM, grad_weight_memory},
         {DNNL_ARG_TO, grad_weight_usr_memory}});
  }
  return std::tuple<at::Tensor, at::Tensor>{grad_weight, grad_bias};
}

} // namespace impl
} // namespace AtenIpexTypeXPU
} // namespace at
