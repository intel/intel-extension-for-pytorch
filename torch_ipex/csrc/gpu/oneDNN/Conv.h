#pragma once

#include <ATen/ATen.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>
#include <core/Quantizer.h>
#include <tensor/Context.h>
#include "Utils.h"
#include "Reorder.h"

#include <oneapi/dnnl/dnnl.hpp>

#ifdef USE_PRIMITIVE_CACHE
#include <oneDNN/LRUCache.h>
#endif


using namespace dnnl;
using namespace xpu::dpcpp;
using namespace at::AtenIpexTypeXPU;

namespace xpu {
namespace oneDNN {

struct ConvAttr {
  static const int64_t kind_with_relu = xpu::oneDNN::with_relu; // 0b01;
  static const int64_t kind_with_sum = xpu::oneDNN::with_sum; // 0b10;
  static const int64_t kind_with_sigmoid = xpu::oneDNN::with_sigmoid; // 0b100;

  ConvAttr() : scale_(1.f), alpha_(0.f), beta_(0.f), oscale_(1.f), attr_(0) {}
  ConvAttr(float scale, float alpha, float beta, float oscale, int64_t attr)
      : scale_(scale), alpha_(alpha), beta_(beta), oscale_(oscale), attr_(attr) {}

  bool with_relu() {
    return attr_ & kind_with_relu;
  }

  bool with_sum() {
    return attr_ & kind_with_sum;
  }

  bool with_sigmoid() {
    return attr_ & kind_with_sigmoid;
  }

  int64_t attr() {
    return attr_;
  }

#ifdef USE_PRIMITIVE_CACHE
  void to_bytes(bytestring& bytes) {
    xpu::dpcpp::to_bytes(bytes, scale_);
    xpu::dpcpp::to_bytes(bytes, alpha_);
    xpu::dpcpp::to_bytes(bytes, beta_);
    xpu::dpcpp::to_bytes(bytes, oscale_);
    xpu::dpcpp::to_bytes(bytes, attr_);
  }
#endif

  float scale_;
  float alpha_;
  float beta_;
  float oscale_;
  int64_t attr_;
};

constexpr int src_batch_size_dim = 0;
constexpr int wgh_dst_channels_dim = 0;

static inline memory::dims conv_dst_tz(
    int64_t ndim,
    IntArrayRef src_tz,
    IntArrayRef wgh_tz,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation) {
  bool has_dilation = dilation.size() > 0;
  memory::dims dst_tz(ndim);
  dst_tz[0] = src_tz[src_batch_size_dim];
  dst_tz[1] = wgh_tz[wgh_dst_channels_dim];
  for (size_t d = 2; d < ndim; ++d) {
    auto dilate = has_dilation ? dilation[d - 2] : 1;
    auto kernel = dilate * (wgh_tz[d] - 1) + 1;
    dst_tz[d] = (src_tz[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return dst_tz;
}

static inline memory::dims compatible_dilation(IntArrayRef &dilation) {
  memory::dims ret = dilation.vec();
  for (auto it = ret.begin(); it != ret.end(); it++) {
    *it -= 1;
  }
  return ret;
}

static inline memory::format_tag
conv_src_fmt(int64_t ndim, bool is_channels_last = false) {
  if (!is_channels_last) {
    return (ndim == 4) ? memory::format_tag::nchw :
                         ((ndim == 5) ? memory::format_tag::ncdhw :
                                        memory::format_tag::undef);
  } else {
    return (ndim == 4) ? memory::format_tag::nhwc :
                         ((ndim == 5) ? memory::format_tag::ndhwc :
                                        memory::format_tag::undef);
  }
}

static inline memory::format_tag
conv_wgh_fmt(int64_t ndim, bool grouped = false, bool is_channels_last = false) {
  if (!is_channels_last) {
    return (ndim == 4) ? (grouped ? memory::format_tag::goihw : memory::format_tag::oihw) :
                         ((ndim == 5) ? (grouped ? memory::format_tag::goidhw : memory::format_tag::oidhw) :
                         memory::format_tag::undef);
  } else {
    return (ndim == 4) ? (grouped ? memory::format_tag::gohwi : memory::format_tag::ohwi) :
                         ((ndim == 5) ? (grouped ? memory::format_tag::godhwi : memory::format_tag::odhwi) :
                         memory::format_tag::undef);
  }
}

static inline memory::dims compatible_wgh_dims(
    int64_t ndim, int64_t groups, int64_t oc, int64_t ic, IntArrayRef wsizes) {
  if (ndim == 4) {
    auto kh = wsizes[2];
    auto kw = wsizes[3];
    return (groups != 1) ? memory::dims({groups, oc / groups, ic / groups, kh, kw})
      : memory::dims({oc, ic, kh, kw});
  } else if (ndim == 5) {
    auto kd = wsizes[2];
    auto kh = wsizes[3];
    auto kw = wsizes[4];
    return (groups != 1) ? memory::dims({groups, oc / groups, ic / groups, kd, kh, kw})
      : memory::dims({oc, ic, kd, kh, kw});
  }

  return {};
}

static at::Tensor convolution(
    at::Tensor& dst,
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& bia,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    ConvAttr attr) {
  auto engine = GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();
  auto ndim = src.ndimension();

  auto dst_tz = conv_dst_tz(
      ndim, src.sizes(), wgh.sizes(), padding, stride, dilation);
  if (!lazy_reorder_enabled() && !dst.defined()) {
    auto dst_opt = src.options();
    if (src.is_quantized()) {
      dst_opt = attr.with_relu() ?
                device(kXPU).dtype(kQUInt8) :
                device(kXPU).dtype(kQInt8);
    }
    if (src.is_contiguous(at::MemoryFormat::ChannelsLast)) {
      dst_opt = dst_opt.memory_format(at::MemoryFormat::ChannelsLast);
    }

    dst = at::empty(dst_tz, dst_opt);
  }

  auto src_data_t = get_onednn_dtype(src);
  auto wei_usr_data_t = get_onednn_dtype(wgh);
  auto wei_data_t = src.is_quantized() ?
                    memory::data_type::s8 :
                    get_onednn_dtype(wgh);
  auto dst_data_t = dst.defined() ? get_onednn_dtype(dst) : src_data_t;
  auto bia_data_t = memory::data_type::f32;
  if (bia.defined()) {
    bia_data_t = (!lazy_reorder_enabled() && src.is_quantized()) ?
                 memory::data_type::s32 :
                 get_onednn_dtype(bia);
  }
  auto usr_bia_data_t = memory::data_type::f32;

  // master wgh
  if (src_data_t == memory::data_type::bf16) {
    wei_data_t = memory::data_type::bf16;
    bia_data_t = memory::data_type::bf16;
    dst_data_t = src_data_t;
  }

  auto ic = src.size(1);
  auto oc = dst_tz[1];

  auto fmt_any = memory::format_tag::any;
  // 4D: n/c/h/w (n/h/w/c)
  // 5D: n/c/d/h/w (n/d/h/w/c)
  auto fmt_src = conv_src_fmt(ndim,
      ndim == 4 ?
      src.is_contiguous(at::MemoryFormat::ChannelsLast) :
      src.is_contiguous(at::MemoryFormat::ChannelsLast3d));
  // 4D: (g)o/i/h/w ((g)o/h/w/i)
  // 5D: (g)o/i/d/h/w ((g)o/d/h/w/i)
  auto fmt_wgh = conv_wgh_fmt(ndim, groups != 1, wgh.size(1) == 1 ? false :
      (wgh.ndimension() == 4 ?
       wgh.is_contiguous(at::MemoryFormat::ChannelsLast) :
       wgh.is_contiguous(at::MemoryFormat::ChannelsLast3d)));
  auto fmt_bia = memory::format_tag::x;

  memory::dims src_tz = src.sizes().vec();
  memory::dims wgh_tz = compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());
  memory::dims bia_tz = {oc};
  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = compatible_dilation(dilation);

  // plain combination
  auto src_md = memory::desc(src_tz, src_data_t, fmt_src);
  auto wgh_md = src.is_contiguous(at::MemoryFormat::ChannelsLast) ?
                memory::desc(wgh_tz, wei_data_t, fmt_any) :
                memory::desc(wgh_tz, wei_data_t, fmt_wgh);
  auto dst_md = memory::desc(dst_tz, dst_data_t, fmt_src);
  auto bia_md = bia.defined() ? memory::desc(bia_tz, bia_data_t, fmt_bia) : memory::desc();

  // block combination
  if (lazy_reorder_enabled()) {
    src_md = memory::desc(src_tz, src_data_t,
        src.is_contiguous(at::MemoryFormat::ChannelsLast) ? fmt_src : fmt_any);
    dst_md = memory::desc(dst_tz, dst_data_t,
        src.is_contiguous(at::MemoryFormat::ChannelsLast) ? fmt_src : fmt_any);
    wgh_md = memory::desc(wgh_tz, wei_data_t, fmt_any);
  }

  auto conv_forward_desc =
      convolution_forward::desc(prop_kind::forward,
                                algorithm::convolution_direct,
                                src_md,
                                wgh_md,
                                bia_md,
                                dst_md,
                                _stride,
                                _dilation,
                                _padding,
                                _padding);

  float src_scale;
  std::vector<float> wgh_scales, conv_scale = {1};
  primitive_attr pattr;
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

    auto dst_scale = attr.oscale_;
    src_scale = src.q_scale();
    conv_scale.clear();
    for (int i = 0; i < wgh_scales.size(); i++) {
      conv_scale.push_back(1.f / (dst_scale / (src_scale * wgh_scales[i])));
    }
    conv_zero_point = static_cast<int>(dst.q_zero_point());

    int mask_ac = 0;
    int mask_conv = wgh_scales.size() > 1 ? 1 << 1 : 0;
    pattr.set_output_scales(mask_conv, conv_scale);
    pattr.set_zero_points(DNNL_ARG_DST, mask_ac, {conv_zero_point});
  }

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key_pd;
  create_key(key_pd, src_md, wgh_md, bia.defined(), dst_data_t,
      _stride, _dilation, _padding, _padding, attr, conv_scale, conv_zero_point);
#endif

  post_ops po;
  if (attr.with_sum())
    po.append_sum(attr.scale_);

  if (attr.with_relu()) {
    po.append_eltwise(1.0, algorithm::eltwise_relu, attr.alpha_, attr.beta_);
  } else if (attr.with_sigmoid()) {
    po.append_eltwise(1.0, algorithm::eltwise_logistic, attr.alpha_, attr.beta_);
  }
  pattr.set_post_ops(po);

  auto conv_forward_pd =
      convolution_forward::primitive_desc(conv_forward_desc, pattr, engine);

  memory::desc src_usr_md, wgh_usr_md, dst_usr_md;
  if (!lazy_reorder_enabled()) {
    src_usr_md = memory::desc(src_tz, src_data_t, fmt_src);
    wgh_usr_md = memory::desc(wgh_tz, wei_usr_data_t, fmt_wgh);
    dst_usr_md = memory::desc(dst_tz, dst_data_t, fmt_src);
  } else {
    auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
    src_usr_md = src_ctx.is_plain()
        ? memory::desc(src_tz, src_data_t, fmt_src)
        : src_ctx.meta();

    auto wgh_ctx = DPCPPTensorContext::get_tensor_ctx(wgh);
    wgh_usr_md = wgh_ctx.is_plain()
        ? memory::desc(wgh_tz, wei_usr_data_t, fmt_wgh)
        : wgh_ctx.meta();

    if (dst.defined()) {
      auto dst_ctx = DPCPPTensorContext::get_tensor_ctx(dst);
      dst_usr_md = dst_ctx.is_plain()
          ? memory::desc(dst_tz, dst_data_t, fmt_src)
          : dst_ctx.meta();
    } else {
      auto expected_dst_md = conv_forward_pd.dst_desc();
      auto plain_dst_md = memory::desc({dst_tz}, dst_data_t, fmt_src);
      auto dst_opt = src.is_contiguous(at::MemoryFormat::ChannelsLast) ?
                     src.options().memory_format(at::MemoryFormat::ChannelsLast) :
                     src.options();
      if (expected_dst_md != plain_dst_md) {
        dst = empty_opaque_tensor(expected_dst_md, dst_opt, c10::nullopt);
      } else {
        // ChannelsLast in block mode
        dst = at::empty(dst_tz, dst_opt);
      }
    }
  }

  Tensor src_, wgh_, dst_ = dst, bia_;

  auto expected_src_md = conv_forward_pd.src_desc();
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

  auto expected_wgh_md = conv_forward_pd.weights_desc();
  auto wgh_m = dpcpp_onednn_memory(wgh_usr_md, engine, wgh.data_ptr());
  if (wgh_usr_md != expected_wgh_md) {
    if (weight_cache_enabled() && src.is_quantized()) {
        QuantizerPtr quantizer;

        if (wgh.is_quantized() && wgh.qscheme() == kPerChannelAffine) {
          quantizer = xpu::dpcpp::make_per_channel_affine_quantizer(
              wgh.q_per_channel_scales(),
              wgh.q_per_channel_zero_points(),
              0,
              kQInt8);
        } else {
          quantizer =
              xpu::dpcpp::make_per_tensor_affine_quantizer(wgh_scales[0], 0, kQInt8);
        }
        wgh_ = empty_opaque_qtensor(expected_wgh_md, c10::nullopt, quantizer);
    } else {
      wgh_ = empty_opaque_tensor(expected_wgh_md, wgh.options(), c10::nullopt);
    }

    wgh_m = dpcpp_onednn_memory(expected_wgh_md, engine, wgh_.data_ptr());
    xpu::oneDNN::reorder(wgh, wgh_);

    if (weight_cache_enabled()) {
      strm.wait();
      auto wgh_opt_ctx = DPCPPTensorContext::release_tensor_ctx(wgh_);
      DPCPPTensorContext::set_tensor_ctx(wgh, std::move(wgh_opt_ctx));
    }
  }

  auto expected_dst_md = conv_forward_pd.dst_desc();
  auto dst_m = dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr());
  if (dst_usr_md != expected_dst_md) {
    if (lazy_reorder_enabled() && dst.is_quantized()) {
      auto quantizer =
          xpu::dpcpp::make_per_tensor_affine_quantizer(dst.q_scale(), dst.q_zero_point(),
          typeMetaToScalarType(dst.options().dtype()));
      dst_ = empty_opaque_qtensor(expected_dst_md, c10::nullopt, quantizer);
    } else {
      dst_ = empty_opaque_tensor(expected_dst_md, dst.options(), c10::nullopt);
    }

    dst_m = dpcpp_onednn_memory(expected_dst_md, engine, dst_.data_ptr());

    if (attr.with_sum())
      xpu::oneDNN::reorder(dst, dst_);
  }

  memory bia_m = memory({{}, bia_data_t, fmt_bia}, engine);
  if (bia.defined()) {
    auto bia_ctx = DPCPPTensorContext::get_tensor_ctx(bia);
    bia_m = bia_ctx.is_plain()
        ? dpcpp_onednn_memory(
              {bia_tz, usr_bia_data_t, fmt_bia}, engine, bia.data_ptr())
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

      if (weight_cache_enabled()) {
        strm.wait();
        // FIXME: thread safty
        auto bia_opt_ctx = DPCPPTensorContext::release_tensor_ctx(bia_);
        DPCPPTensorContext::set_tensor_ctx(bia, std::move(bia_opt_ctx));
      }
    }
  }

#ifdef USE_PRIMITIVE_CACHE
  auto conv_forward = fetch_or_create_m<convolution_forward>(key_pd, conv_forward_pd);
#else
  auto conv_forward = convolution_forward(conv_forward_pd);
#endif

  DPCPP_ONEDNN_EXEC(
      conv_forward,
      strm,
      {{DNNL_ARG_SRC,       src_m},
       {DNNL_ARG_WEIGHTS,   wgh_m},
       {DNNL_ARG_BIAS,      bia_m},
       {DNNL_ARG_DST,       dst_m}}
  );

  if (lazy_reorder_enabled() && dst_.data_ptr() != dst.data_ptr()) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(dst_);
    DPCPPTensorContext::set_tensor_ctx(dst, std::move(blk_ctx));
  }

  return dst;
}

}}
