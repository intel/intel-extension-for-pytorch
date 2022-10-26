#pragma once

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/record_function.h>
#include <core/MemoryFormat.h>
#include <core/TensorImplUtils.h>

#include <oneDNN/Runtime.h>
#include <quantized/Quantizer.h>
#include <runtime/Utils.h>
#include <tensor/Context.h>
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

constexpr int src_batch_size_dim = 0;
constexpr int wgh_dst_channels_dim = 0;

static inline memory::dims conv_dst_tz(
    int64_t ndim,
    IntArrayRef src_tz,
    IntArrayRef wgh_tz,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation) {
  bool has_dilation = dilation.size() > 0;
  memory::dims dst_tz(ndim);
  dst_tz[0] = src_tz[src_batch_size_dim];
  dst_tz[1] = wgh_tz[wgh_dst_channels_dim];
  for (size_t d = 2; d < ndim; ++d) {
    auto dilate = has_dilation ? dilation[d - 2] : 1;
    auto kernel = dilate * (wgh_tz[d] - 1) + 1;
    dst_tz[d] =
        (src_tz[d] +
         (padding_front_top_left[d - 2] + padding_back_bottom_right[d - 2]) -
         kernel) /
            stride[d - 2] +
        1;
  }
  return dst_tz;
}

static inline memory::dims compatible_dilation(IntArrayRef& dilation) {
  memory::dims ret = dilation.vec();
  for (auto it = ret.begin(); it != ret.end(); it++) {
    *it -= 1;
  }
  return ret;
}

static inline memory::format_tag conv_src_fmt(
    const int64_t ndim,
    const bool is_channels_last = false) {
  if (!is_channels_last) {
    return (ndim == 3)
        ? memory::format_tag::ncw
        : ((ndim == 4) ? memory::format_tag::nchw
                       : ((ndim == 5) ? memory::format_tag::ncdhw
                                      : memory::format_tag::undef));
  } else {
    return (ndim == 3)
        ? memory::format_tag::nwc
        : ((ndim == 4) ? memory::format_tag::nhwc
                       : ((ndim == 5) ? memory::format_tag::ndhwc
                                      : memory::format_tag::undef));
  }
}

static inline memory::format_tag conv_wgh_fmt(
    const int64_t ndim,
    const bool grouped = false,
    const bool is_channels_last = false) {
  if (!is_channels_last) {
    return (ndim == 3)
        ? (grouped ? memory::format_tag::goiw : memory::format_tag::oiw)
        : (ndim == 4)
            ? (grouped ? memory::format_tag::goihw : memory::format_tag::oihw)
            : ((ndim == 5) ? (grouped ? memory::format_tag::goidhw
                                      : memory::format_tag::oidhw)
                           : memory::format_tag::undef);
  } else {
    return (ndim == 3)
        ? (grouped ? memory::format_tag::gowi : memory::format_tag::owi)
        : (ndim == 4)
            ? (grouped ? memory::format_tag::gohwi : memory::format_tag::ohwi)
            : ((ndim == 5) ? (grouped ? memory::format_tag::godhwi
                                      : memory::format_tag::odhwi)
                           : memory::format_tag::undef);
  }
}

static inline memory::dims compatible_wgh_dims(
    const int64_t ndim,
    const int64_t groups,
    const int64_t oc,
    const int64_t ic,
    const IntArrayRef wsizes) {
  if (ndim == 3) {
    auto kw = wsizes[2];
    return (groups != 1) ? memory::dims({groups, oc / groups, ic / groups, kw})
                         : memory::dims({oc, ic, kw});
  } else if (ndim == 4) {
    auto kh = wsizes[2];
    auto kw = wsizes[3];
    return (groups != 1)
        ? memory::dims({groups, oc / groups, ic / groups, kh, kw})
        : memory::dims({oc, ic, kh, kw});
  } else if (ndim == 5) {
    auto kd = wsizes[2];
    auto kh = wsizes[3];
    auto kw = wsizes[4];
    return (groups != 1)
        ? memory::dims({groups, oc / groups, ic / groups, kd, kh, kw})
        : memory::dims({oc, ic, kd, kh, kw});
  }

  return {};
}

static inline bool onednn_conv_use_channels_last(
    const at::Tensor& src,
    const at::Tensor& weight) {
  // Convolution modules, unlike binary p-wise operator, have
  // channels last as the dominating memory format. If both
  // src and weight are in contiguous memory format, the
  // operator produces output in contiguous memory format.
  // Otherwise, output will be in channels last memory format.
  return (is_smf_channels_last(src) || is_smf_channels_last(weight));
}

static convolution_forward::primitive_desc get_convolution_pd(
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& bia,
    const IntArrayRef padding,
    const IntArrayRef stride,
    IntArrayRef dilation,
    const int64_t groups) {
  auto src_data_t = get_onednn_dtype(src);
  auto wei_data_t = get_onednn_dtype(wgh);
  auto bia_data_t =
      bia.defined() ? get_onednn_dtype(bia) : memory::data_type::undef;
  auto dst_data_t = src_data_t;

  auto ndim = src.ndimension();
  memory::dims dst_tz = conv_dst_tz(
      ndim, src.sizes(), wgh.sizes(), padding, padding, stride, dilation);
  auto ic = src.size(1);
  auto oc = dst_tz[1];
  memory::dims src_tz = src.sizes().vec();
  memory::dims wgh_tz = compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());
  memory::dims bia_tz = {oc};

  auto fmt_any = memory::format_tag::any;
  auto src_md = memory::desc(src_tz, src_data_t, fmt_any);
  auto wgh_md = memory::desc(wgh_tz, wei_data_t, fmt_any);
  auto bia_md = bia.defined() ? memory::desc(bia_tz, bia_data_t, fmt_any)
                              : memory::desc();
  auto dst_md = memory::desc(dst_tz, dst_data_t, fmt_any);

  auto conv_forward_desc = convolution_forward::desc(
      prop_kind::forward,
      algorithm::convolution_direct,
      src_md,
      wgh_md,
      bia_md,
      dst_md,
      stride.vec(),
      compatible_dilation(dilation),
      padding.vec(),
      padding.vec());

  primitive_attr pattr;
#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(scratchpad_mode::user);
#endif

  return convolution_forward::primitive_desc(
      conv_forward_desc,
      pattr,
      GpuEngineManager::Instance().get_engine({kXPU, current_device()}));
}

static at::Tensor convolution(
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
  auto is_onednn_layout_suggested = using_onednn_layout_for_conv(src);
  if (!is_onednn_layout_suggested && !dst.defined()) {
    auto dst_opt = src.options();
    if (src.is_quantized()) {
      dst_opt = attr.get_dst_dtype();
    }
    if (onednn_conv_use_channels_last(src, wgh)) {
      TORCH_CHECK(
          3 == ndim || 4 == ndim || 5 == ndim,
          "convolution only supports 3D, 4D, 5D tensor");
      dst_opt = dst_opt.memory_format(get_cl_tag_by_ndim(ndim));
    }

    dst = at::empty(dst_tz, dst_opt);
  }
  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  auto src_data_t = src_ctx.is_plain() ? get_onednn_dtype_include_double(src)
                                       : src_ctx.meta().data_type();
  auto wei_usr_data_t = get_onednn_dtype_include_double(wgh);
  auto wei_data_t = src.is_quantized() ? memory::data_type::s8
                                       : get_onednn_dtype_include_double(wgh);
  auto dst_data_t =
      dst.defined() ? get_onednn_dtype_include_double(dst) : src_data_t;

  // if src is quant, set bia data type to f32
  // if src is not quant, get user demanded data type
  auto bia_data_t = bia.defined() && src.is_quantized()
      ? memory::data_type::f32
      : bia.defined() ? get_onednn_dtype_include_double(bia)
                      : memory::data_type::undef;

  if (memory::data_type::bf16 == src_data_t && bia.defined()) {
    // if src data type is bf16 and bia is defined, bia data type must be bf16
    // or f32
    TORCH_CHECK(
        memory::data_type::f32 == bia_data_t ||
        memory::data_type::bf16 == bia_data_t);
  }

  auto ic = src.size(1);
  auto oc = dst_tz[1];

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

  memory::dims src_tz = src.sizes().vec();
  memory::dims wgh_tz = compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());
  memory::dims bia_tz = {oc};
  memory::dims _stride = stride.vec();
  memory::dims _padding_front_top_left = padding_front_top_left.vec();
  memory::dims _padding_back_bottom_right = padding_back_bottom_right.vec();
  memory::dims _dilation = compatible_dilation(dilation);

  // plain combination
  auto src_md = memory::desc(src_tz, src_data_t, fmt_src);
  auto wgh_md = onednn_conv_use_channels_last(src, wgh)
      ? memory::desc(wgh_tz, wei_data_t, fmt_any)
      : memory::desc(wgh_tz, wei_data_t, fmt_wgh);
  auto dst_md = memory::desc(dst_tz, dst_data_t, fmt_src);
  auto bia_md = bia.defined() ? memory::desc(bia_tz, bia_data_t, fmt_bia)
                              : memory::desc();

  // block combination
  if (is_onednn_layout_suggested) {
    // In blocked format scenario, oneDNN accept the src in plain format
    // when src ic = 3
    if (ic == 3) {
      src_md = memory::desc(src_tz, src_data_t, fmt_src);
    } else {
      src_md = memory::desc(src_tz, src_data_t, fmt_any);
    }
    dst_md = memory::desc(dst_tz, dst_data_t, fmt_any);
    wgh_md = memory::desc(wgh_tz, wei_data_t, fmt_any);
  }

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
  if (src.is_quantized()) {
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

    src_scale =
        (src_data_t == memory::data_type::u8 && src.q_zero_point() == 128)
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
  }

  std::unordered_map<int, memory> args;
  post_ops po;
  attr.extract_post_ops(po, dst);
  pattr.set_post_ops(po);

#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

  if (src_data_t == memory::data_type::f32) {
    pattr.set_fpmath_mode(xpu::oneDNN::get_onednn_fpmath_mode());
  }

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

  memory::desc src_usr_md, wgh_usr_md, dst_usr_md;

  // block weight when NHWC
  auto wgh_ctx = DPCPPTensorContext::get_tensor_ctx(wgh);
  wgh_usr_md = wgh_ctx.is_plain()
      ? memory::desc(wgh_tz, wei_usr_data_t, fmt_wgh)
      : wgh_ctx.meta();

  if (!is_onednn_layout_suggested) {
    src_usr_md = memory::desc(src_tz, src_data_t, fmt_src);
    dst_usr_md = memory::desc(dst_tz, dst_data_t, fmt_src);
  } else {
    auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
    src_usr_md = src_ctx.is_plain() ? memory::desc(src_tz, src_data_t, fmt_src)
                                    : src_ctx.meta();

    if (dst.defined()) {
      auto dst_ctx = DPCPPTensorContext::get_tensor_ctx(dst);
      dst_usr_md = dst_ctx.is_plain()
          ? memory::desc(dst_tz, dst_data_t, fmt_src)
          : dst_ctx.meta();
    } else {
      auto expected_dst_md = conv_fwd_pd.dst_desc();
      auto plain_dst_md = memory::desc({dst_tz}, dst_data_t, fmt_src);
      auto mem_fmt = get_cl_tag_by_ndim(ndim);
      auto dst_opt = onednn_conv_use_channels_last(src, wgh)
          // src.options() is just used to fill dst_opt. can also be
          // any other tensor to do this
          ? src.options().memory_format(mem_fmt)
          : src.options();
      if (expected_dst_md != plain_dst_md) {
        dst = empty_opaque_tensor(expected_dst_md, dst_opt, c10::nullopt);
      } else {
        // ChannelsLast in block mode
        dst = at::empty(dst_tz, dst_opt);
      }
    }
  }

  Tensor src_, wgh_, dst_ = dst, bia_;

  auto expected_src_md = conv_fwd_pd.src_desc();
  memory src_m = dpcpp_onednn_memory(src_usr_md, engine, src.data_ptr());
  if (src_usr_md != expected_src_md) {
    // avoid reorder in case of, [n][C][1][1][16c] <==> [n][c][1][1]
    if (src.sizes().size() == 4 && src.size(2) == 1 && src.size(3) == 1) {
      src_m = dpcpp_onednn_memory(expected_src_md, engine, src.data_ptr());
    } else {
      src_ = empty_opaque_tensor(expected_src_md, src.options(), c10::nullopt);
      src_m = dpcpp_onednn_memory(expected_src_md, engine, src_.data_ptr());
      xpu::oneDNN::reorder(src, src_);
    }
  }

  auto weight_cache_optimization = [&]() {
    bool onoff = false;
    onoff |= is_onednn_layout_suggested;
    onoff |= onednn_conv_use_channels_last(src, wgh);
    onoff &= !at::GradMode::is_enabled();
    return onoff;
  }();

  auto expected_wgh_md = conv_fwd_pd.weights_desc();
  auto wgh_m = dpcpp_onednn_memory(wgh_usr_md, engine, wgh.data_ptr());
  if (wgh_usr_md != expected_wgh_md) {
    if (weight_cache_optimization && src.is_quantized()) {
      QuantizerPtr quantizer;

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
    } else {
      wgh_ = empty_opaque_tensor(expected_wgh_md, wgh.options(), c10::nullopt);
    }

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
  auto expected_dst_md = conv_fwd_pd.dst_desc();
  auto dst_m = dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr());
  if (dst_usr_md != expected_dst_md) {
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

    if (attr.with_sum())
      xpu::oneDNN::reorder(dst, dst_);
  }

  if (attr.with_binary())
    attr.construct_post_binary(conv_fwd_pd, po, expected_dst_md, args);

  memory bia_m = memory({{}, bia_data_t, fmt_bia}, engine);
  if (bia.defined()) {
    auto bia_ctx = DPCPPTensorContext::get_tensor_ctx(bia);

    bia_m = bia_ctx.is_plain()
        ? dpcpp_onednn_memory(
              {bia_tz, bia_data_t, fmt_bia}, engine, bia.data_ptr())
        : dpcpp_onednn_memory({bia_ctx.meta()}, engine, bia.data_ptr());

    if (bia_ctx.is_plain() && src.is_quantized()) {
      std::vector<float> bia_scale;
      for (int i = 0; i < wgh_scales.size(); i++) {
        bia_scale.push_back(1.f / (src_scale * wgh_scales[i] / 1.f));
      }

      int mask = wgh_scales.size() > 1 ? ONEDNN_SCALES_MASK_BY_CHANNEL(0) : 0;
      auto reorder_attr = xpu::oneDNN::ReorderAttr();
      reorder_attr.set_dst_sc_and_zp(mask, bia_scale, 0, {0});

      bia_ = empty_opaque_tensor(bia_md, bia.options(), c10::nullopt);
      bia_m = dpcpp_onednn_memory(bia_md, engine, bia_.data_ptr());
      xpu::oneDNN::reorder(bia, bia_, reorder_attr);

// Following is for saving bias correctly.
// TODO: Need a general solution for bias caching
#ifndef BUILD_JIT_QUANTIZATION_SAVE
      if (weight_cache_optimization) {
        strm.wait();
        // FIXME: thread safty
        auto bia_opt_ctx = DPCPPTensorContext::release_tensor_ctx(bia_);
        DPCPPTensorContext::set_tensor_ctx(bia, std::move(bia_opt_ctx));
      }
#endif
    }
  }

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

static std::tuple<at::Tensor, at::Tensor> convolution_backward_weights(
    const at::Tensor& diff_dst,
    const at::Tensor& src,
    IntArrayRef diff_wgh_aten_tz,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool with_bias) {
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto ndim = diff_dst.ndimension();
  TORCH_CHECK(
      3 == ndim || 4 == ndim || 5 == ndim,
      "convolution bwd wgh only supports 3D, 4D, 5D tensor");
  auto smf = onednn_conv_use_channels_last(src, diff_dst)
      ? get_cl_tag_by_ndim(ndim)
      : at::MemoryFormat::Contiguous;

  Tensor diff_bia;
  if (with_bias) {
    diff_bia = at::empty({diff_dst.size(1)}, diff_dst.options());
  }

  auto diff_wgh = at::empty(diff_wgh_aten_tz, diff_dst.options(), smf);
  if (src.numel() == 0) {
    return std::tuple<at::Tensor, at::Tensor>{diff_wgh, diff_bia};
  }

  auto ic = src.size(1);
  auto oc = diff_dst.size(1);

  memory::data_type diff_dst_dt = get_onednn_dtype_include_double(diff_dst);
  memory::data_type src_dt = get_onednn_dtype_include_double(src);
  TORCH_CHECK(
      diff_dst_dt == src_dt,
      "convolution bwd_wb need same dtype for src and diff_dst");
  memory::data_type wgh_dt = src_dt;
  memory::data_type dst_dt = src_dt;
  memory::data_type bia_dt = src_dt;

  memory::format_tag any_fmt = memory::format_tag::any;
  memory::format_tag src_fmt =
      conv_src_fmt(ndim, onednn_conv_use_channels_last(src, diff_dst));
  memory::format_tag wgh_fmt = conv_wgh_fmt(
      ndim, groups != 1, onednn_conv_use_channels_last(src, diff_dst));
  memory::format_tag dst_fmt = src_fmt;
  memory::format_tag bia_fmt = memory::format_tag::x;

  memory::dims src_tz = src.sizes().vec();
  memory::dims wgh_tz =
      compatible_wgh_dims(ndim, groups, oc, ic, diff_wgh.sizes());
  memory::dims bia_tz = {oc};
  memory::dims dst_tz = diff_dst.sizes().vec();
  dst_tz[0] = src.size(0); // set n

  memory::dims _stride = stride.vec();
  memory::dims _padding_front_top_left = padding_front_top_left.vec();
  memory::dims _padding_back_bottom_right = padding_back_bottom_right.vec();
  memory::dims _dilation = compatible_dilation(dilation);

  auto src_md = memory::desc(src_tz, src_dt, src_fmt);
  auto wgh_md = onednn_conv_use_channels_last(src, diff_dst)
      ? memory::desc(wgh_tz, wgh_dt, any_fmt)
      : memory::desc(wgh_tz, wgh_dt, wgh_fmt);
  auto dst_md = memory::desc(dst_tz, dst_dt, dst_fmt);
  auto bia_md =
      with_bias ? memory::desc(bia_tz, bia_dt, bia_fmt) : memory::desc();

  auto conv_fwd_desc = with_bias ? convolution_forward::desc(
                                       prop_kind::forward,
                                       algorithm::convolution_direct,
                                       src_md,
                                       wgh_md,
                                       bia_md,
                                       dst_md,
                                       _stride,
                                       _dilation,
                                       _padding_front_top_left,
                                       _padding_back_bottom_right)
                                 : convolution_forward::desc(
                                       prop_kind::forward,
                                       algorithm::convolution_direct,
                                       src_md,
                                       wgh_md,
                                       dst_md,
                                       _stride,
                                       _dilation,
                                       _padding_front_top_left,
                                       _padding_back_bottom_right);

  primitive_attr pattr;
  if (src_dt == memory::data_type::f32) {
    pattr.set_fpmath_mode(xpu::oneDNN::get_onednn_fpmath_mode());
  }

#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

  auto conv_fwd_pd =
      convolution_forward::primitive_desc(conv_fwd_desc, pattr, engine);

  if (Settings::I().is_onednn_layout_enabled()) {
    src_md = memory::desc(src_tz, src_dt, any_fmt);
    wgh_md = memory::desc(wgh_tz, wgh_dt, any_fmt);
    dst_md = memory::desc(dst_tz, dst_dt, any_fmt);
    bia_md = with_bias ? memory::desc(bia_tz, bia_dt, bia_fmt) : memory::desc();
  }

  auto conv_bwd_w_desc = with_bias ? convolution_backward_weights::desc(
                                         algorithm::convolution_direct,
                                         src_md,
                                         wgh_md,
                                         bia_md,
                                         dst_md,
                                         _stride,
                                         _dilation,
                                         _padding_front_top_left,
                                         _padding_back_bottom_right)
                                   : convolution_backward_weights::desc(
                                         algorithm::convolution_direct,
                                         src_md,
                                         wgh_md,
                                         dst_md,
                                         _stride,
                                         _dilation,
                                         _padding_front_top_left,
                                         _padding_back_bottom_right);

  auto conv_bwd_w_pd = convolution_backward_weights::primitive_desc(
      conv_bwd_w_desc, pattr, engine, conv_fwd_pd);

  memory src_usr_m, diff_dst_usr_m, diff_wgh_usr_m;
  if (!Settings::I().is_onednn_layout_enabled()) {
    src_usr_m =
        dpcpp_onednn_memory({src_tz, src_dt, src_fmt}, engine, src.data_ptr());

    diff_dst_usr_m = dpcpp_onednn_memory(
        {dst_tz, diff_dst_dt, dst_fmt}, engine, diff_dst.data_ptr());

    diff_wgh_usr_m = dpcpp_onednn_memory(
        {wgh_tz, wgh_dt, wgh_fmt}, engine, diff_wgh.data_ptr());
  } else {
    auto src_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src);
    src_usr_m = src_ctx.is_plain()
        ? dpcpp_onednn_memory({src_tz, src_dt, src_fmt}, engine, src.data_ptr())
        : dpcpp_onednn_memory({src_ctx.meta()}, engine, src.data_ptr());

    auto diff_dst_ctx =
        at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(diff_dst);
    diff_dst_usr_m = diff_dst_ctx.is_plain()
        ? dpcpp_onednn_memory(
              {dst_tz, diff_dst_dt, dst_fmt}, engine, diff_dst.data_ptr())
        : dpcpp_onednn_memory(
              {diff_dst_ctx.meta()}, engine, diff_dst.data_ptr());

    auto diff_wgh_ctx =
        at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(diff_wgh);
    diff_wgh_usr_m = diff_wgh_ctx.is_plain()
        ? dpcpp_onednn_memory(
              {wgh_tz, wgh_dt, wgh_fmt}, engine, diff_wgh.data_ptr())
        : dpcpp_onednn_memory(
              {diff_wgh_ctx.meta()}, engine, diff_wgh.data_ptr());
  }

  Tensor src_;
  auto expected_src_md = conv_bwd_w_pd.src_desc();
  auto src_m = src_usr_m;
  if (src_usr_m.get_desc() != expected_src_md) {
    src_ = empty_opaque_tensor(expected_src_md, src.options(), c10::nullopt);
    src_m = dpcpp_onednn_memory(expected_src_md, engine, src_.data_ptr());
    xpu::oneDNN::reorder(src, src_);
  }

  Tensor diff_dst_;
  auto expected_diff_dst_md = conv_bwd_w_pd.diff_dst_desc();
  auto diff_dst_m = diff_dst_usr_m;
  if (diff_dst_usr_m.get_desc() != expected_diff_dst_md) {
    diff_dst_ = empty_opaque_tensor(
        expected_diff_dst_md, diff_dst.options(), c10::nullopt);
    diff_dst_m =
        dpcpp_onednn_memory(expected_diff_dst_md, engine, diff_dst_.data_ptr());
    xpu::oneDNN::reorder(diff_dst, diff_dst_);
  }

  Tensor diff_wgh_;
  auto expected_diff_wgh_md = conv_bwd_w_pd.diff_weights_desc();
  auto diff_wgh_m = diff_wgh_usr_m;
  if (diff_wgh_usr_m.get_desc() != expected_diff_wgh_md) {
    diff_wgh_ =
        empty_opaque_tensor(expected_diff_wgh_md, diff_wgh.options(), smf);
    diff_wgh_m =
        dpcpp_onednn_memory(expected_diff_wgh_md, engine, diff_wgh_.data_ptr());
  }

#ifdef USE_SCRATCHPAD_MODE
  int scratchpad_size = conv_bwd_w_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, src.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = dnnl::memory(
      conv_bwd_w_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
#endif

  auto conv_bwd_w = dnnl::convolution_backward_weights(conv_bwd_w_pd);
  if (with_bias) {
    memory diff_bia_m = dpcpp_onednn_memory(
        {bia_tz, bia_dt, bia_fmt}, engine, diff_bia.data_ptr());

    DPCPP_ONEDNN_EXEC(
        conv_bwd_w,
        strm,
        {
            {DNNL_ARG_DIFF_DST, diff_dst_m},
            {DNNL_ARG_SRC, src_m},
            {DNNL_ARG_DIFF_WEIGHTS, diff_wgh_m},
            {DNNL_ARG_DIFF_BIAS, diff_bia_m},
#ifdef USE_SCRATCHPAD_MODE
            {DNNL_ARG_SCRATCHPAD, scratchpad_m},
#endif
        });
  } else {
    DPCPP_ONEDNN_EXEC(
        conv_bwd_w,
        strm,
        {
            {DNNL_ARG_DIFF_DST, diff_dst_m},
            {DNNL_ARG_SRC, src_m},
            {DNNL_ARG_DIFF_WEIGHTS, diff_wgh_m},
#ifdef USE_SCRATCHPAD_MODE
            {DNNL_ARG_SCRATCHPAD, scratchpad_m},
#endif
        });
  }

  if (diff_wgh_m.get_desc() != diff_wgh_usr_m.get_desc()) {
    // diff_wgh_ contains the result of gw
    // backward, while it is blk format. In
    // training mode, plain gw output is
    // expected for sgd update regardless of
    // onednn_layout_enabled or not. Thus, we
    // need one additional reorder here to make
    // diff_wgh plain.
    xpu::oneDNN::reorder(diff_wgh_, diff_wgh);
  }

  return std::tuple<at::Tensor, at::Tensor>{diff_wgh, diff_bia};
}

} // namespace oneDNN
} // namespace xpu
