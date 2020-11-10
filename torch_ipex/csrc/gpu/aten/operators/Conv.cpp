#include <ATen/quantized/QTensorImpl.h>
#include <ATen/ipex_type_dpcpp_customized.h>
#include <ATen/quantized/Quantizer.h>
#include <core/DPCPPUtils.h>
#include <core/Quantizer.h>
#include <core/Runtime.h>
#include <core/TensorImplUtils.h>
#include <tensor/Context.h>

#include <utils/ParamUtils.h>

#include "Conv.h"

using namespace dnnl;
using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

at::Tensor convolution(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    conv_attr_t attr) {
  auto engine = GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto ndim = input.ndimension();
  auto output_tz = conv_output_size(
      ndim, input.sizes(), weight.sizes(), padding, stride, dilation);
  if (!lazy_reorder_enabled() && !output.defined()) {
    auto out_dt = attr.with_relu() ? device(kXPU).dtype(kQUInt8) : device(kXPU).dtype(kQInt8);
    output = at::empty(output_tz, input.is_quantized() ? out_dt : input.options());
  }

  auto src_data_t = dt_to_dnnl(input.scalar_type());
  auto wei_usr_data_t = dt_to_dnnl(weight.scalar_type());
  auto wei_data_t = input.is_quantized() ? dnnl::memory::data_type::s8
    : dt_to_dnnl(weight.scalar_type());
  auto dst_data_t = output.defined() ? dt_to_dnnl(output.scalar_type()) : src_data_t;
  auto bias_data_t = dnnl::memory::data_type::f32;
  if (bias.defined()) {
    bias_data_t = (!lazy_reorder_enabled() && input.is_quantized()) ? dnnl::memory::data_type::s32
      : dt_to_dnnl(bias.scalar_type());
  }
  auto usr_bias_data_t = dnnl::memory::data_type::f32;

  auto format_any = memory::format_tag::any;
  auto format_input = conv_input_fmt(ndim);
  auto format_weight = conv_weight_fmt(ndim, groups != 1);
  auto format_bias = memory::format_tag::x;

  auto ic = input.size(1);
  auto oc = output_tz[1];
  memory::dims input_tz = input.sizes().vec();
  memory::dims weight_tz = compatible_weight_dims(ndim, groups, oc, ic, weight.sizes());
  memory::dims bias_tz = {oc};
  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = compatible_dilation(dilation);

  auto input_md = memory::desc(input_tz, src_data_t, format_any);
  auto weight_md = memory::desc(weight_tz, wei_data_t, format_any);
  auto output_md = memory::desc(output_tz, dst_data_t, format_any);
  auto bias_md = bias.defined() ? memory::desc(bias_tz, bias_data_t, format_bias) : memory::desc();

  auto conv_forward_desc = convolution_forward::desc(
      prop_kind::forward, algorithm::convolution_direct,
      input_md, weight_md, bias_md, output_md,
      _stride, _dilation, _padding, _padding);

  std::vector<float> weight_scales;
  if (input.is_quantized()) {
    auto weight_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(weight);
    if (lazy_reorder_enabled() && !weight_ctx.is_plain()) {
      weight_scales = weight_ctx.scales();
    } else {
      if (weight.is_quantized()) {
        if (weight.qscheme() == kPerTensorAffine) {
          weight_scales.push_back(static_cast<float>(weight.q_scale()));
        } else {
          for (int i = 0; i < oc; i++) {
            weight_scales.push_back(weight.q_per_channel_scales()[i].item<float>());
          }
        }
      }
    }
  }

  primitive_attr pattr;
  float in_scale;
  std::vector<float> conv_scale;
  int conv_zero_point;
  if (input.is_quantized()) {
    auto out_scale = attr.oscale_;
    in_scale = input.q_scale();
    for (int i = 0; i < weight_scales.size(); i++) {
      conv_scale.push_back(1.f / (out_scale / (in_scale * weight_scales[i])));
    }
    conv_zero_point = static_cast<int>(output.q_zero_point());
    int mask_ac = 0;
    int mask_conv = weight_scales.size() > 1 ? 1 << 1 : 0;
    pattr.set_output_scales(mask_conv, conv_scale);
    pattr.set_zero_points(DNNL_ARG_DST, mask_ac, {conv_zero_point});
  }

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, input_md, weight_md, bias.defined(), dst_data_t,
      _stride, _dilation, _padding, _padding, attr, conv_scale, conv_zero_point);
#endif

  post_ops po;
  if (attr.with_sum()) {
    if (input.is_quantized()) {
      float with_sum_out_scale = attr.scale_;
      po.append_sum(with_sum_out_scale);
    } else {
      po.append_sum(attr.scale_);
    }
  }

  if (attr.with_relu()) {
    po.append_eltwise(1.0, algorithm::eltwise_relu, attr.alpha_, attr.beta_);
  } else if (attr.with_sigmoid()) {
    po.append_eltwise(1.0, algorithm::eltwise_logistic, attr.alpha_, attr.beta_);
  }
  pattr.set_post_ops(po);

  auto conv_forward_pd = convolution_forward::primitive_desc(conv_forward_desc, pattr, engine);
  memory input_usr_memory, weight_usr_memory, output_usr_memory;
  if (!lazy_reorder_enabled()) {
    input_usr_memory = dpcpp_onednn_memory(
        {input_tz, src_data_t, format_input}, engine, input.data_ptr());

    weight_usr_memory = dpcpp_onednn_memory(
        {weight_tz, wei_usr_data_t, format_weight}, engine, weight.data_ptr());

    output_usr_memory = dpcpp_onednn_memory(
        {output_tz, dst_data_t, format_input}, engine, output.data_ptr());
  } else {
    auto input_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(input);
    input_usr_memory = input_ctx.is_plain()
        ? dpcpp_onednn_memory({input_tz, src_data_t, format_input}, engine, input.data_ptr())
        : dpcpp_onednn_memory({input_ctx.meta()}, engine, input.data_ptr());

    auto weight_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(weight);
    weight_usr_memory = weight_ctx.is_plain()
        ? dpcpp_onednn_memory({weight_tz, wei_usr_data_t, format_input}, engine, weight.data_ptr())
        : dpcpp_onednn_memory({weight_ctx.meta()}, engine, weight.data_ptr());

    if (output.defined()) {
      auto output_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(output);
      output_usr_memory = output_ctx.is_plain()
          ? dpcpp_onednn_memory({output_tz, dst_data_t, format_input}, engine, output.data_ptr())
          : dpcpp_onednn_memory({output_ctx.meta()}, engine, output.data_ptr());
    } else {
      auto expected_output_md = conv_forward_pd.dst_desc();
      auto plain_output_md = memory::desc({output_tz}, dst_data_t, format_input);
      if (expected_output_md != plain_output_md) {
        output = empty_opaque_tensor(expected_output_md, input.options(), c10::nullopt);
        output_usr_memory = dpcpp_onednn_memory(expected_output_md, engine, output.data_ptr());
      } else {
        output = at::empty(output_tz, input.options());
        output_usr_memory = dpcpp_onednn_memory(plain_output_md, engine, output.data_ptr());
      }
    }
  }

  auto expected_input_md = conv_forward_pd.src_desc();
  auto input_memory = input_usr_memory;
  Tensor input_;
  if (input_usr_memory.get_desc() != expected_input_md) {
    // avoid reorder in case of, [n][C][1][1][16c] <==> [n][c][1][1]
    if (input.sizes().size() == 4 && input.size(2) == 1 && input.size(3) == 1) {
      input_memory = dpcpp_onednn_memory(expected_input_md, engine, input.data_ptr());
    } else {
      auto item_num = static_cast<int64_t>(expected_input_md.get_size() / input.itemsize());
      input_ = at::AtenIpexTypeXPU::empty({item_num}, input.options(), c10::nullopt);
      input_memory = dpcpp_onednn_memory(expected_input_md, engine, input_.data_ptr());
      DPCPP_ONEDNN_EXEC(
          reorder(input_usr_memory, input_memory),
          strm,
          {{DNNL_ARG_FROM, input_usr_memory},
          {DNNL_ARG_TO, input_memory}});
    }
  }

  auto expected_weight_md = conv_forward_pd.weights_desc();
  auto weight_memory = weight_usr_memory;
  Tensor weight_;
  if (weight_usr_memory.get_desc() != expected_weight_md) {
    Tensor weight_opt;
    if (weight_cache_enabled()) {
      if (input.is_quantized()) {
        QuantizerPtr quantizer;
        if (weight.is_quantized() && weight.qscheme() == kPerChannelAffine) {
          quantizer = make_per_channel_affine_quantizer(
              weight.q_per_channel_scales(),
              weight.q_per_channel_zero_points(),
              0,
              kQInt8);
        } else {
          quantizer = make_per_tensor_affine_quantizer(weight_scales[0], 0, kQInt8);
        }
        weight_opt = empty_opaque_qtensor(expected_weight_md, c10::nullopt, quantizer);
      } else {
        weight_opt = empty_opaque_tensor(expected_weight_md, weight.options(), c10::nullopt);
      }
      weight_memory = dpcpp_onednn_memory(expected_weight_md, engine, weight_opt.data_ptr());
    } else {
      auto item_num = static_cast<int64_t>(expected_weight_md.get_size() / weight.itemsize());
      weight_ = at::AtenIpexTypeXPU::empty({item_num}, weight.options(), c10::nullopt);
      weight_memory = dpcpp_onednn_memory(expected_weight_md, engine, weight_.data_ptr());
    }

    DPCPP_ONEDNN_EXEC(
        reorder(weight_usr_memory, weight_memory),
        strm,
        {{DNNL_ARG_FROM, weight_usr_memory},
        {DNNL_ARG_TO, weight_memory}});

    if (weight_cache_enabled()) {
      strm.wait();
      auto weight_opt_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::release_tensor_ctx(weight_opt);
      at::AtenIpexTypeXPU::DPCPPTensorContext::set_tensor_ctx(weight, std::move(weight_opt_ctx));
    }
  }

  auto expected_output_md = conv_forward_pd.dst_desc();
  auto output_memory = output_usr_memory;
  Tensor output_;
  if (output_usr_memory.get_desc() != expected_output_md) {
    if (lazy_reorder_enabled()) {
      if (output.is_quantized()) {
        auto quantizer = make_per_tensor_affine_quantizer(output.q_scale(),
            output.q_zero_point(), typeMetaToScalarType(output.options().dtype()));
        output_ = empty_opaque_qtensor(expected_output_md, c10::nullopt, quantizer);
      } else {
        output_ = empty_opaque_tensor(expected_output_md, output.options(), c10::nullopt);
      }
    } else {
      auto item_num = static_cast<int64_t>(expected_output_md.get_size() / output.itemsize());
      output_ = at::AtenIpexTypeXPU::empty({item_num}, output.options(), c10::nullopt);
    }
    output_memory = dpcpp_onednn_memory(expected_output_md, engine, output_.data_ptr());
    if (attr.with_sum()) {
      DPCPP_ONEDNN_EXEC(
          reorder(output_usr_memory, output_memory),
          strm,
          {{DNNL_ARG_FROM, output_usr_memory},
          {DNNL_ARG_TO, output_memory}});
    }
  }

  Tensor bias_;
  Tensor bias_opt;
  memory bias_memory = memory({{}, bias_data_t, format_bias}, engine);
  if (bias.defined()) {
    auto bias_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(bias);
    auto bias_usr_memory = bias_ctx.is_plain()
        ? dpcpp_onednn_memory({bias_tz, usr_bias_data_t, format_bias}, engine, bias.data_ptr())
        : dpcpp_onednn_memory({bias_ctx.meta()}, engine, bias.data_ptr());

    bias_memory = bias_usr_memory;
    if (bias_ctx.is_plain() && input.is_quantized()) {
      if (weight_cache_enabled()) {
        bias_opt = empty_opaque_tensor(bias_md, bias.options(), c10::nullopt);
        bias_memory = dpcpp_onednn_memory(bias_md, engine, bias_opt.data_ptr());
      } else {
        auto item_num = static_cast<int64_t>(bias_md.get_size() / bias.itemsize());
        bias_ = at::AtenIpexTypeXPU::empty({item_num}, bias.options(), c10::nullopt);
        bias_memory = dpcpp_onednn_memory(bias_md, engine, bias_.data_ptr());
      }

      primitive_attr attr;
      std::vector<float> bias_scale;
      for (int i = 0; i < weight_scales.size(); i++) {
        bias_scale.push_back(1.f / (in_scale * weight_scales[i] / 1.f));
      }

      int mask = weight_scales.size() > 1 ? 1 << 0 : 0;
      attr.set_output_scales(mask, bias_scale);
      attr.set_zero_points(DNNL_ARG_DST, 0, {0}); // TODO:Asymmetric

      DPCPP_ONEDNN_EXEC(
          reorder(bias_usr_memory, bias_memory, attr),
          strm,
          {{DNNL_ARG_FROM, bias_usr_memory},
          {DNNL_ARG_TO, bias_memory}});

      if (weight_cache_enabled()) {
        strm.wait();
        // FIXME: thread safty
        auto bias_opt_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::release_tensor_ctx(bias_opt);
        at::AtenIpexTypeXPU::DPCPPTensorContext::set_tensor_ctx(bias, std::move(bias_opt_ctx));
      }
    }
  }

#ifdef USE_PRIMITIVE_CACHE
  auto conv_forward = fetch_or_create_m<convolution_forward>(key, conv_forward_pd);
#else
  auto conv_forward = convolution_forward(conv_forward_pd);
#endif

  DPCPP_ONEDNN_EXEC(
      conv_forward,
      strm,
      {{DNNL_ARG_SRC, input_memory},
       {DNNL_ARG_WEIGHTS, weight_memory},
       {DNNL_ARG_BIAS, bias_memory},
       {DNNL_ARG_DST, output_memory}});

  if (!lazy_reorder_enabled() && output_memory != output_usr_memory) {
    DPCPP_ONEDNN_EXEC(
        reorder(output_memory, output_usr_memory),
        strm,
        {{DNNL_ARG_FROM, output_memory},
        {DNNL_ARG_TO, output_usr_memory}});
  } else if (lazy_reorder_enabled() && output_memory != output_usr_memory) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(output_);
    DPCPPTensorContext::set_tensor_ctx(output, std::move(blk_ctx));
  }

  return output;
}

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
  auto data_grad = dt_to_dnnl(grad_output.scalar_type());
  auto weight_t = dt_to_dnnl(weight.scalar_type());
  auto bias_t = dnnl::memory::data_type::f32;
  auto format_any = memory::format_tag::any;
  auto format_input = conv_input_fmt(ndim);
  auto format_weight = conv_weight_fmt(ndim, groups != 1);

  memory::dims input_tz = grad_input.sizes().vec();
  memory::dims weight_tz = compatible_weight_dims(ndim, groups, oc, ic, weight.sizes());
  memory::dims bias_tz = {oc};
  memory::dims output_tz = grad_output.sizes().vec();
  output_tz[0] = grad_input.size(0); // set n

  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = compatible_dilation(dilation);

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
      {weight_tz, weight_t, format_weight}, engine, weight.data_ptr());

    grad_input_usr_memory = dpcpp_onednn_memory(
      {input_tz, data_grad, format_input}, engine, grad_input.data_ptr());
  } else {
    auto grad_output_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(grad_output);
    grad_output_usr_memory = grad_output_ctx.is_plain()
        ? dpcpp_onednn_memory({output_tz, data_grad, format_input}, engine, grad_output.data_ptr())
        : dpcpp_onednn_memory({grad_output_ctx.meta()}, engine, grad_output.data_ptr());

    auto weight_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(weight);
    weight_usr_memory = weight_ctx.is_plain()
        ? dpcpp_onednn_memory({weight_tz, weight_t, format_weight}, engine, weight.data_ptr())
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
        reorder(grad_output_usr_memory, grad_output_memory),
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
        reorder(weight_usr_memory, weight_memory),
        strm,
        {{DNNL_ARG_FROM, weight_usr_memory},
        {DNNL_ARG_TO, weight_memory}});
  }

  Tensor grad_input_;
  auto expected_grad_input_md = conv_backward_data_pd.diff_src_desc();
  auto grad_input_memory = grad_input_usr_memory;
  if (grad_input_memory.get_desc() != expected_grad_input_md) {
    auto item_num = static_cast<int64_t>(expected_grad_input_md.get_size() / grad_input.itemsize());
    grad_input_ = at::AtenIpexTypeXPU::empty({item_num}, grad_input.options(), c10::nullopt);
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
        reorder(grad_input_memory, grad_input_usr_memory),
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
  auto data_grad = dt_to_dnnl(grad_output.scalar_type());
  auto weight_t = dt_to_dnnl(grad_output.scalar_type());
  auto bias_t = memory::data_type::f32;
  auto format_any = memory::format_tag::any;
  auto format_input = conv_input_fmt(ndim);
  auto format_weight = conv_weight_fmt(ndim, groups != 1);
  auto format_bias = memory::format_tag::x;

  memory::dims input_tz = input.sizes().vec();
  memory::dims weight_tz = compatible_weight_dims(ndim, groups, oc, ic, grad_weight.sizes());
  memory::dims bias_tz = {oc};
  memory::dims output_tz = grad_output.sizes().vec();
  output_tz[0] = input.size(0); // set n

  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = compatible_dilation(dilation);

  auto input_md = memory::desc(input_tz, data_grad, format_any);
  auto weight_md = memory::desc(weight_tz, weight_t, format_any);
  auto output_md = memory::desc(output_tz, data_grad, format_any);
  auto bias_md = bias_defined ? memory::desc(bias_tz, bias_t, format_any) : memory::desc();

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
        {weight_tz, weight_t, format_weight}, engine, grad_weight.data_ptr());
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
        ? dpcpp_onednn_memory({weight_tz, weight_t, format_weight}, engine, grad_weight.data_ptr())
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
        reorder(input_usr_memory, input_memory),
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
        reorder(grad_output_usr_memory, grad_output_memory),
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
        ? dpcpp_onednn_memory({bias_tz, bias_t, format_bias}, engine, grad_bias.data_ptr())
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

  if (!lazy_reorder_enabled() && grad_weight_memory != grad_weight_usr_memory) {
    DPCPP_ONEDNN_EXEC(
        reorder(grad_weight_memory, grad_weight_usr_memory),
        strm,
        {{DNNL_ARG_FROM, grad_weight_memory},
        {DNNL_ARG_TO, grad_weight_usr_memory}});
  } else if (lazy_reorder_enabled() && grad_weight_memory != grad_weight_usr_memory) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(grad_weight_);
    DPCPPTensorContext::set_tensor_ctx(grad_weight, std::move(blk_ctx));
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
    conv_attr_t attr) {
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

  Tensor output_ = convolution(
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
    conv_attr_t attr) {
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
  conv_attr_t attr(
      scale.to<float>(),
      alpha.to<float>(),
      beta.to<float>(),
      1.f,
      conv_attr_t::kind_with_sum);
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
  conv_attr_t attr(
      scale.to<float>(),
      alpha.to<float>(),
      beta.to<float>(),
      1.f,
      conv_attr_t::kind_with_relu | conv_attr_t::kind_with_sum);
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
  conv_attr_t attr(
      scale.to<float>(),
      alpha.to<float>(),
      beta.to<float>(),
      1.f,
      conv_attr_t::kind_with_relu);
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
  conv_attr_t attr(
      scale.to<float>(),
      alpha.to<float>(),
      beta.to<float>(),
      1.f,
      conv_attr_t::kind_with_sigmoid);
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
      conv_attr_t());
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
  if (output_mask[1] || output_mask[2]) {
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

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

} // namespace AtenIpexTypeXPU
} // namespace at

