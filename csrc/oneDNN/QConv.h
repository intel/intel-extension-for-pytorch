#pragma once

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/record_function.h>
#include <core/MemoryFormat.h>
#include <core/TensorImplUtils.h>

#include <oneDNN/Conv.h>
#include <oneDNN/Runtime.h>
#include <quantized/Quantizer.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <utils/LRUCache.h>
#include "Attr.h"
#include "Reorder.h"
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace at::AtenIpexTypeXPU;
using namespace at::AtenIpexTypeQuantizedXPU;

namespace xpu {
namespace oneDNN {

static memory::desc get_quantized_src_md(
    const at::Tensor& src,
    const at::Tensor& wgh,
    int64_t ndim,
    bool is_onednn_layout_suggested) {
  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  auto src_data_t = src_ctx.is_plain() ? get_onednn_dtype_include_double(src)
                                       : src_ctx.meta().data_type();
  memory::dims src_tz = src.sizes().vec();

  auto fmt_any = memory::format_tag::any;
  auto fmt_src = conv_src_fmt(ndim, onednn_conv_use_channels_last(src, wgh));
  auto src_md = memory::desc(src_tz, src_data_t, fmt_src);

  auto ic = src.size(1);
  // block combination
  if (is_onednn_layout_suggested) {
    // In blocked format scenario, oneDNN accept the src in plain format
    // when src ic = 3
    if (ic == 3) {
      src_md = memory::desc(src_tz, src_data_t, fmt_src);
    } else {
      src_md = memory::desc(src_tz, src_data_t, fmt_any);
    }
  }
  return src_md; // TODO: use std::move here?
}

static memory::desc get_quantized_dst_md(
    at::Tensor& dst,
    const at::Tensor& src,
    const at::Tensor& wgh,
    int64_t ndim,
    const memory::dims& dst_tz,
    bool is_onednn_layout_suggested,
    Attr attr) {
  auto fmt_src = conv_src_fmt(ndim, onednn_conv_use_channels_last(src, wgh));

  // dst always be created in QConv.cpp, aka dst is defined
  TORCH_CHECK(dst.defined(), "Quantized convlution should always define dst");

  auto dst_data_t = get_onednn_dtype_include_double(dst);

  auto dst_md = memory::desc(dst_tz, dst_data_t, fmt_src);
  return dst_md;
}

static memory::desc get_quantized_bia_md(
    const at::Tensor& bia,
    memory::dims dst_tz) {
  auto bia_data_t = bia.defined() ? get_onednn_dtype_include_double(bia)
                                  : memory::data_type::undef;

  auto oc = dst_tz[1];
  memory::dims bia_tz = {oc};
  auto fmt_bia = memory::format_tag::x;
  auto bia_md = bia.defined() ? memory::desc(bia_tz, bia_data_t, fmt_bia)
                              : memory::desc();
  return bia_md;
}

static memory::desc get_quantized_wgh_md(
    const at::Tensor& src,
    const at::Tensor& wgh,
    memory::dims wgh_tz,
    int64_t groups,
    int64_t ndim) {
  auto wei_data_t = memory::data_type::s8;
  auto fmt_any = memory::format_tag::any;
  auto fmt_wgh =
      conv_wgh_fmt(ndim, groups != 1, onednn_conv_use_channels_last(src, wgh));
  auto wgh_md = onednn_conv_use_channels_last(src, wgh)
      ? memory::desc(wgh_tz, wei_data_t, fmt_any)
      : memory::desc(wgh_tz, wei_data_t, fmt_wgh);
  return wgh_md;
}

static memory::desc get_quantized_primitive_src_md(
    const Tensor& src,
    const Tensor& wgh,
    int64_t ndim,
    bool is_onednn_layout_suggested) {
  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  auto src_data_t = src_ctx.is_plain() ? get_onednn_dtype_include_double(src)
                                       : src_ctx.meta().data_type();
  memory::dims src_tz = src.sizes().vec();
  memory::desc src_usr_md;
  auto fmt_src = conv_src_fmt(ndim, onednn_conv_use_channels_last(src, wgh));
  if (!is_onednn_layout_suggested) {
    src_usr_md = memory::desc(src_tz, src_data_t, fmt_src);
  } else {
    auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
    src_usr_md = src_ctx.is_plain() ? memory::desc(src_tz, src_data_t, fmt_src)
                                    : src_ctx.meta();
  }
  return src_usr_md;
}

static memory::desc get_quantized_primitive_dst_md(
    at::Tensor& dst,
    const at::Tensor& src,
    const at::Tensor& wgh,
    memory::dims dst_tz,
    int64_t ndim,
    convolution_forward::primitive_desc conv_fwd_pd,
    bool is_onednn_layout_suggested) {
  memory::desc dst_usr_md;

  auto fmt_src = conv_src_fmt(ndim, onednn_conv_use_channels_last(src, wgh));
  auto dst_data_t = get_onednn_dtype_include_double(dst);
  if (!is_onednn_layout_suggested) {
    dst_usr_md = memory::desc(dst_tz, dst_data_t, fmt_src);
  } else {
    auto dst_ctx = DPCPPTensorContext::get_tensor_ctx(dst);
    dst_usr_md = dst_ctx.is_plain() ? memory::desc(dst_tz, dst_data_t, fmt_src)
                                    : dst_ctx.meta();
  }
  return dst_usr_md;
}

static memory::desc get_quantized_primitive_wgh_md(
    const at::Tensor& wgh,
    memory::dims wgh_tz,
    memory::data_type wei_usr_data_t,
    memory::format_tag fmt_wgh) {
  memory::desc wgh_usr_md;
  auto wgh_ctx = DPCPPTensorContext::get_tensor_ctx(wgh);
  wgh_usr_md = wgh_ctx.is_plain()
      ? memory::desc(wgh_tz, wei_usr_data_t, fmt_wgh)
      : wgh_ctx.meta();
  return wgh_usr_md;
}

static memory get_quantized_src_memory(
    const at::Tensor& src,
    at::Tensor& src_,
    memory::desc& src_usr_md,
    convolution_forward::primitive_desc& conv_fwd_pd,
    dnnl::engine& engine) {
  auto expected_src_md = conv_fwd_pd.src_desc();
  memory src_m = dpcpp_onednn_memory(src_usr_md, engine, src.data_ptr());
  if (src_usr_md != expected_src_md) {
    // avoid reorder in case of, [n][C][1][1][16c] <==> [n][c][1][1]
    if (src.sizes().size() == 4 && src.size(2) == 1 && src.size(3) == 1) {
      src_m = dpcpp_onednn_memory(expected_src_md, engine, src.data_ptr());
    } else {
      // TODO: empty opaque_qtensor?
      src_ = empty_opaque_tensor(expected_src_md, src.options(), c10::nullopt);
      src_m = dpcpp_onednn_memory(expected_src_md, engine, src_.data_ptr());
      xpu::oneDNN::reorder(src, src_);
    }
  }
  return src_m;
}

static memory get_quantized_wgh_memory(
    const at::Tensor& wgh,
    at::Tensor& wgh_,
    std::vector<float>& wgh_scales,
    memory::desc& wgh_usr_md,
    convolution_forward::primitive_desc& conv_fwd_pd,
    dnnl::engine& engine,
    dnnl::stream& strm,
    bool weight_cache_optimization) {
  auto expected_wgh_md = conv_fwd_pd.weights_desc();
  auto wgh_m = dpcpp_onednn_memory(wgh_usr_md, engine, wgh.data_ptr());

  if (wgh_usr_md != expected_wgh_md) {
    QuantizerPtr quantizer;
    // TODO: Assert wgh is quantized?
    if (wgh.is_quantized() && wgh.qscheme() == kPerChannelAffine) {
      quantizer = dpcpp_make_per_channel_affine_quantizer(
          wgh.q_per_channel_scales(),
          wgh.q_per_channel_zero_points(),
          0,
          kQInt8);
    } else {
      quantizer =
          dpcpp_make_per_tensor_affine_quantizer(wgh_scales[0], 0, kQInt8);
    }
    wgh_ = empty_opaque_qtensor(expected_wgh_md, c10::nullopt, quantizer);

    wgh_m = dpcpp_onednn_memory(expected_wgh_md, engine, wgh_.data_ptr());
    auto reshaped_wgh = wgh;
    // reshape for group convolution weight
    if (wgh_.ndimension() == 5 && wgh.ndimension() == 4) {
      reshaped_wgh = share_storage_and_set_strided_as(
          wgh,
          wgh_.sizes(),
          /*compatible with different strides of weight (including contiguous,
             channels_last and non-contiguous) */
          compatible_groups_conv_strides(wgh, wgh_),
          c10::nullopt);
    }
    xpu::oneDNN::reorder(reshaped_wgh, wgh_);

    if (weight_cache_optimization) {
      strm.wait();
      auto wgh_opt_ctx = DPCPPTensorContext::release_tensor_ctx(wgh_);
      DPCPPTensorContext::set_tensor_ctx(wgh, std::move(wgh_opt_ctx));
    }
  }
  return wgh_m;
}

static memory get_quantized_dst_memory(
    at::Tensor& dst,
    at::Tensor& dst_,
    memory::desc& dst_usr_md,
    bool is_onednn_layout_suggested,
    convolution_forward::primitive_desc& conv_fwd_pd,
    Attr& attr,
    dnnl::engine& engine) {
  auto expected_dst_md = conv_fwd_pd.dst_desc();
  auto dst_m = dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr());
  if (dst_usr_md != expected_dst_md) {
    // TODO: Why condition is is_onednn_layout_suggested && dst.is_quantized()?
    // if is_onednn_layout_sugeesed=False, will use non-quantized
    // empty_opaque_tensor
    if (is_onednn_layout_suggested && dst.is_quantized()) {
      auto quantizer = dpcpp_make_per_tensor_affine_quantizer(
          (get_onednn_dtype_include_double(dst) == memory::data_type::u8 &&
           dst.q_zero_point() == 128)
              ? dst.q_scale() / 2
              : dst.q_scale(),
          0,
          typeMetaToScalarType(dst.options().dtype()));
      dst_ = empty_opaque_qtensor(expected_dst_md, c10::nullopt, quantizer);
    } else {
      dst_ = empty_opaque_tensor(expected_dst_md, dst.options(), c10::nullopt);
    }

    dst_m = dpcpp_onednn_memory(expected_dst_md, engine, dst_.data_ptr());

    // TODO: Why with_sum need reorder?
    if (attr.with_sum())
      xpu::oneDNN::reorder(dst, dst_);
  }
  return dst_m;
}

static memory get_quantized_bia_memory(
    const Tensor& bia,
    memory::dims bia_tz,
    memory::data_type bia_data_t,
    memory::format_tag fmt_bia,
    dnnl::engine& engine) {
  memory bia_m = memory({{}, bia_data_t, fmt_bia}, engine);
  if (bia.defined()) {
    auto bia_ctx = DPCPPTensorContext::get_tensor_ctx(bia);

    bia_m = bia_ctx.is_plain()
        ? dpcpp_onednn_memory(
              {bia_tz, bia_data_t, fmt_bia}, engine, bia.data_ptr())
        : dpcpp_onednn_memory({bia_ctx.meta()}, engine, bia.data_ptr());
  }
  return bia_m;
}

static at::Tensor quantized_convolution(
    at::Tensor& dst,
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& bia,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    Attr& attr) {
  // TODO: Check src is quantized
  TORCH_CHECK(
      !bia.defined(), "QConv only supports binary_add post-op for bias");
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto ndim = src.ndimension();
  auto dst_tz = conv_dst_tz(
      ndim,
      src.sizes(),
      wgh.sizes(),
      padding_front_top_left,
      padding_back_bottom_right,
      stride,
      dilation);
  auto ic = src.size(1);
  auto oc = dst_tz[1];

  memory::dims src_tz = src.sizes().vec();
  memory::dims bia_tz = {oc};
  memory::dims wgh_tz = compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());

  auto wei_usr_data_t = get_onednn_dtype_include_double(wgh);
  auto wei_data_t = src.is_quantized() ? memory::data_type::s8
                                       : get_onednn_dtype_include_double(wgh);
  auto bia_data_t = bia.defined() ? get_onednn_dtype_include_double(bia)
                                  : memory::data_type::undef;

  auto fmt_any = memory::format_tag::any;
  // 3D: n/c/w (n/w/c)
  // 4D: n/c/h/w (n/h/w/c)
  // 5D: n/c/d/h/w (n/d/h/w/c)
  auto fmt_src = conv_src_fmt(ndim, onednn_conv_use_channels_last(src, wgh));
  // 3D: (g)o/i/w ((g)o/w/i)
  // 4D: (g)o/i/h/w ((g)o/h/w/i)
  // 5D: (g)o/i/d/h/w ((g)o/d/h/w/i)
  auto fmt_wgh =
      conv_wgh_fmt(ndim, groups != 1, onednn_conv_use_channels_last(src, wgh));
  auto fmt_bia = memory::format_tag::x;

  auto is_onednn_layout_suggested = using_onednn_layout_for_conv(src);

  auto src_md =
      get_quantized_src_md(src, wgh, ndim, is_onednn_layout_suggested);
  auto dst_md = get_quantized_dst_md(
      dst, src, wgh, ndim, dst_tz, is_onednn_layout_suggested, attr);
  auto wgh_md = get_quantized_wgh_md(src, wgh, wgh_tz, groups, ndim);
  auto bia_md = get_quantized_bia_md(bia, dst_tz);

  memory::dims _stride = stride.vec();
  memory::dims _padding_front_top_left = padding_front_top_left.vec();
  memory::dims _padding_back_bottom_right = padding_back_bottom_right.vec();
  memory::dims _dilation = compatible_dilation(dilation);

  auto conv_fwd_desc = convolution_forward::desc(
      prop_kind::forward,
      algorithm::convolution_direct,
      src_md,
      wgh_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right);

  primitive_attr pattr;
  float src_scale;
  std::vector<float> wgh_scales, conv_scale = {1};
  int conv_zero_point = 0;
  auto wgh_ctx = DPCPPTensorContext::get_tensor_ctx(wgh);
  if (!wgh_ctx.is_plain()) {
    wgh_scales = wgh_ctx.scales();
  } else {
    if (wgh.qscheme() == kPerTensorAffine) {
      wgh_scales.push_back(static_cast<float>(wgh.q_scale()));
    } else {
      for (int i = 0; i < oc; i++) {
        wgh_scales.push_back(wgh.q_per_channel_scales()[i].item<float>());
      }
    }
  }

  // TODO: scale setting in separate functions
  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  auto src_data_t = src_ctx.is_plain() ? get_onednn_dtype_include_double(src)
                                       : src_ctx.meta().data_type();
  src_scale = (src_data_t == memory::data_type::u8 && src.q_zero_point() == 128)
      ? src.q_scale() / 2
      : src.q_scale();
  conv_scale.clear();
  /* Note: [Convolution requantization]
      Suppose y = w * x. The * refer to convolution operation, and y, w, x are
      dtype of FP32.
      Then we have
        y_int / y_sc =  (w_int / w_sc) * (x_int / x_sc) =>
        y_int = [y_sc / (w_sc x x_sc)] (w_int * x_int).
      The y_sc / (w_sc x x_sc) is requantization scale, which is also the
      conv_scale in following line.
      Inversion is required due to scale_onednn = 1  / scale_torch */
  /*The requantization will be performed in Attr.h with appending post op
   * linear to adjust the scale/zeropoint*/
  for (int i = 0; i < wgh_scales.size(); i++) {
    conv_scale.push_back(1.f / (1.f / (src_scale * wgh_scales[i])));
  }
  conv_zero_point = static_cast<int>(0);
  int mask_ac = 0;
  int mask_conv = wgh_scales.size() > 1 ? 1 << 1 : 0;
  pattr.set_output_scales(mask_conv, conv_scale);
  pattr.set_zero_points(DNNL_ARG_DST, mask_ac, {conv_zero_point});

  std::unordered_map<int, memory> args;
  post_ops po;
  attr.extract_post_ops(po, dst);
  pattr.set_post_ops(po);

#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key_pd;
  create_key(
      key_pd,
      src_md,
      wgh_md,
      bia.defined(),
      dst_data_t,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      attr,
      conv_scale,
      conv_zero_point);
#endif

  auto conv_fwd_pd =
      convolution_forward::primitive_desc(conv_fwd_desc, pattr, engine);

  auto wgh_usr_md =
      get_quantized_primitive_wgh_md(wgh, wgh_tz, wei_usr_data_t, fmt_wgh);
  auto src_usr_md = get_quantized_primitive_src_md(
      src, wgh, ndim, is_onednn_layout_suggested);
  auto dst_usr_md = get_quantized_primitive_dst_md(
      dst, src, wgh, dst_tz, ndim, conv_fwd_pd, is_onednn_layout_suggested);

  Tensor src_, wgh_, bia_;
  Tensor dst_ = dst;

  memory src_m =
      get_quantized_src_memory(src, src_, src_usr_md, conv_fwd_pd, engine);

  auto weight_cache_optimization = [&]() {
    bool onoff = false;
    onoff |= is_onednn_layout_suggested;
    onoff |= onednn_conv_use_channels_last(src, wgh);
    onoff &= !at::GradMode::is_enabled();
    return onoff;
  }();

  memory wgh_m = get_quantized_wgh_memory(
      wgh,
      wgh_,
      wgh_scales,
      wgh_usr_md,
      conv_fwd_pd,
      engine,
      strm,
      weight_cache_optimization);
  memory dst_m = get_quantized_dst_memory(
      dst,
      dst_,
      dst_usr_md,
      is_onednn_layout_suggested,
      conv_fwd_pd,
      attr,
      engine);

  // TODO: Why post binary use expected_dst_md?
  auto expected_dst_md = conv_fwd_pd.dst_desc();
  if (attr.with_binary())
    attr.construct_post_binary(conv_fwd_pd, po, expected_dst_md, args);

  memory bia_m =
      get_quantized_bia_memory(bia, bia_tz, bia_data_t, fmt_bia, engine);

#ifdef USE_PRIMITIVE_CACHE
  auto conv_forward =
      fetch_or_create_m<convolution_forward>(key_pd, conv_fwd_pd);
#else
  auto conv_forward = convolution_forward(conv_fwd_pd);
#endif

  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_WEIGHTS, wgh_m});
  args.insert({DNNL_ARG_BIAS, bia_m});
  args.insert({DNNL_ARG_DST, dst_m});

#ifdef USE_SCRATCHPAD_MODE
  int scratchpad_size = conv_fwd_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, src.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = dpcpp_onednn_memory(
      conv_fwd_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});
#endif

  DPCPP_ONEDNN_EXEC(conv_forward, strm, args);
  if (is_onednn_layout_suggested && dst_.data_ptr() != dst.data_ptr()) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(dst_);
    DPCPPTensorContext::set_tensor_ctx(dst, std::move(blk_ctx));
  }

  return dst;
}

} // namespace oneDNN
} // namespace xpu
