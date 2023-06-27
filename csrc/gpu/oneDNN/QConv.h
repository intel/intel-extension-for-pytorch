#pragma once

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/record_function.h>
#include <core/MemoryFormat.h>

#include <oneDNN/Conv.h>
#include <oneDNN/Runtime.h>
#include <quantized/QUtils.h>
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

static std::tuple<memory::desc, memory::desc, memory::desc> qconv_get_usr_md(
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& dst,
    int64_t groups,
    int memory_layout) {
  // create memory desc from the src/wgh/dst tensors
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md;
  auto ndim = src.ndimension();
  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  auto fmt_src =
      conv_src_fmt(ndim, memory_layout == MEMORY_LAYOUT_FOR_CONV::ChannelsLast);

  if (src_ctx.is_plain()) {
    auto src_tz = src.sizes().vec();
    auto src_data_t = (src.scalar_type() == at::kQInt8 || is_opaque_u8(src))
        ? memory::data_type::s8
        : memory::data_type::u8;
    src_usr_md = memory::desc(src_tz, src_data_t, fmt_src);
  } else {
    src_usr_md = src_ctx.meta();
  }

  auto dst_ctx = DPCPPTensorContext::get_tensor_ctx(dst);
  if (dst_ctx.is_plain()) {
    auto dst_tz = dst.sizes().vec();
    auto dst_data_t = get_onednn_dtype(dst);
    dst_usr_md = memory::desc(dst_tz, dst_data_t, fmt_src);
  } else {
    dst_usr_md = dst_ctx.meta();
  }

  auto wgh_ctx = DPCPPTensorContext::get_tensor_ctx(wgh);
  if (wgh_ctx.is_plain()) {
    auto ic = src.size(1);
    auto oc = dst.size(1);
    auto wei_data_t = memory::data_type::s8;
    memory::dims wgh_tz =
        compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());
    auto fmt_wgh = conv_wgh_fmt(
        ndim,
        groups != 1,
        memory_layout == MEMORY_LAYOUT_FOR_CONV::ChannelsLast);
    wgh_usr_md = memory::desc(wgh_tz, wei_data_t, fmt_wgh);
  } else {
    wgh_usr_md = wgh_ctx.meta();
  }

  return {src_usr_md, wgh_usr_md, dst_usr_md};
}

static std::tuple<memory::desc, memory::desc, memory::desc> qconv_get_blocked_md(
    const at::Tensor& src,
    memory::desc src_usr_md,
    memory::desc wgh_usr_md,
    memory::desc dst_usr_md) {
  // create memory desc for conv primitive and query the blocked format
  memory::desc src_md, wgh_md, dst_md;
  auto fmt_any = memory::format_tag::any;
  src_md = src.size(1) == 3
      ? src_usr_md
      : memory::desc(
            src_usr_md.get_dims(), src_usr_md.get_data_type(), fmt_any);
  wgh_md = memory::desc(wgh_usr_md.get_dims(), memory::data_type::s8, fmt_any);
  dst_md =
      memory::desc(dst_usr_md.get_dims(), dst_usr_md.get_data_type(), fmt_any);

  return {src_md, wgh_md, dst_md};
}

static std::tuple<memory::desc, memory::desc, memory::desc> qconv_get_plain_md(
    memory::desc src_usr_md,
    memory::desc wgh_usr_md,
    memory::desc dst_usr_md,
    memory::dims wgh_tz,
    bool is_channels_last_suggested) {
  // create memory desc for conv primitive and query the blocked format
  memory::desc src_md, wgh_md, dst_md;
  src_md = src_usr_md;
  dst_md = dst_usr_md;
  if (is_channels_last_suggested) {
    // TODO: remove this path when oneDNN fix the accuracy issue.
    // in ChannelsLast senario, fmt_wgh should be nhwc instead of any
    auto fmt_any = memory::format_tag::any;
    auto wei_data_t = memory::data_type::s8;
    wgh_md = memory::desc(wgh_tz, wei_data_t, fmt_any);
  } else {
    wgh_md = wgh_usr_md;
  }
  return {src_md, wgh_md, dst_md};
}

static memory qconv_get_expected_src_memory(
    const at::Tensor& src,
    at::Tensor& src_blocked,
    memory::desc& src_usr_md,
    memory::desc& expected_src_md,
    dnnl::engine& engine,
    bool load_from_cache) {
  memory src_m;
  if (src_usr_md != expected_src_md) {
    // avoid reorder in case of, [n][C][1][1][16c] <==> [n][c][1][1]
    if (src.sizes().size() == 4 && src.size(2) == 1 && src.size(3) == 1) {
      src_m = dpcpp_onednn_memory(expected_src_md, engine, src.data_ptr());
    } else {
      // See Note [empty_opaque_tensor for qtensor creation]
      src_blocked =
          empty_opaque_tensor(expected_src_md, src.options(), c10::nullopt);
      src_m =
          dpcpp_onednn_memory(expected_src_md, engine, src_blocked.data_ptr());
      xpu::oneDNN::reorder(src, src_blocked);
    }
  } else {
    src_m = dpcpp_onednn_memory(src_usr_md, engine, src.data_ptr());
    src_blocked = src;
  }
  return src_m;
}

static memory qconv_get_expected_wgh_memory(
    const at::Tensor& wgh,
    at::Tensor& wgh_blocked,
    memory::desc& wgh_usr_md,
    memory::desc& expected_wgh_md,
    std::vector<float>& wgh_scales,
    dnnl::engine& engine,
    bool weight_cache_optimization) {
  memory wgh_m;
  if (wgh_usr_md != expected_wgh_md) {
    wgh_blocked =
        empty_opaque_tensor(expected_wgh_md, wgh.options(), c10::nullopt);
    wgh_m =
        dpcpp_onednn_memory(expected_wgh_md, engine, wgh_blocked.data_ptr());

    auto reshaped_wgh = wgh;
    // reshape for group convolution weight
    if (wgh_blocked.ndimension() > wgh.ndimension()) {
      reshaped_wgh = share_storage_and_set_strided_as(
          wgh,
          wgh_blocked.sizes(),
          /*compatible with different strides of weight (including contiguous,
             channels_last and non-contiguous) */
          compatible_groups_conv_strides(wgh, wgh_blocked.sizes().vec()),
          c10::nullopt);
    }
    xpu::oneDNN::reorder(reshaped_wgh, wgh_blocked);

    if (weight_cache_optimization) {
      auto wgh_opt_ctx = DPCPPTensorContext::release_tensor_ctx(wgh_blocked);
      wgh_opt_ctx.set_aten_meta(
          {reshaped_wgh.sizes().vec(), reshaped_wgh.strides().vec()});
      DPCPPTensorContext::set_tensor_ctx(wgh, std::move(wgh_opt_ctx));
    }
  } else {
    wgh_m = dpcpp_onednn_memory(wgh_usr_md, engine, wgh.data_ptr());
    wgh_blocked = wgh;
  }
  return wgh_m;
}

static memory qconv_get_blocked_dst_memory(
    at::Tensor& dst,
    at::Tensor& dst_blocked,
    memory::desc& dst_usr_md,
    memory::desc& expected_dst_md,
    Attr& attr,
    dnnl::engine& engine) {
  memory dst_m;
  if (dst_usr_md != expected_dst_md) {
    auto quantizer = dpcpp_make_per_tensor_affine_quantizer(
        (get_onednn_dtype(dst) == memory::data_type::u8 &&
         dst.q_zero_point() == 128)
            ? dst.q_scale() / 2
            : dst.q_scale(),
        0,
        typeMetaToScalarType(dst.options().dtype()));
    dst_blocked =
        empty_opaque_qtensor(expected_dst_md, c10::nullopt, quantizer);
    dst_m =
        dpcpp_onednn_memory(expected_dst_md, engine, dst_blocked.data_ptr());

    if (attr.with_sum())
      xpu::oneDNN::reorder(dst, dst_blocked);
  } else {
    dst_m = dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr());
    dst_blocked = dst;
  }
  return dst_m;
}

static at::Tensor quantized_convolution(
    at::Tensor& dst,
    const at::Tensor& src,
    const at::Tensor& wgh,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    Attr& attr) {
  // TODO: Check src is quantized
  TORCH_CHECK(src.is_quantized(), "QConv only supports quantized src");
  TORCH_CHECK(wgh.is_quantized(), "QConv only supports quantized wgh");
  TORCH_CHECK(dst.is_quantized(), "QConv only supports quantized dst");
  auto ndim = src.ndimension();
  TORCH_CHECK(
      3 == ndim || 4 == ndim || 5 == ndim,
      "convolution only supports 3D, 4D, 5D tensor");
  TORCH_CHECK(dst.defined(), "Quantized convlution should always define dst");
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  // create usr_md for tensors, and md for conv primitive
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md;
  auto memory_layout_for_conv = get_memory_layout_for_conv(src, wgh);
  bool is_onednn_layout_suggested =
      memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked;
  bool is_channels_last_suggested =
      memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::ChannelsLast;
  // input tensors config
  memory::dims src_dims = src.sizes().vec();
  memory::dims wgh_dims = wgh.sizes().vec();
  auto src_data_t = (src.scalar_type() == at::kQInt8 || is_opaque_u8(src))
      ? memory::data_type::s8
      : memory::data_type::u8;
  auto dst_data_t = get_onednn_dtype_include_double(dst);
  // conv config
  memory::dims _stride = stride.vec();
  memory::dims _padding_front_top_left = padding_front_top_left.vec();
  memory::dims _padding_back_bottom_right = padding_back_bottom_right.vec();
  memory::dims _dilation = compatible_dilation(dilation);
  lru_key_t key_primitive;
  post_ops po;
  // extract post ops
  attr.extract_post_ops(po, dst);
  // set conv primitive scale and zero_point
  std::vector<float> wgh_scales, conv_scale = {1};
  int mask_ac = 0, mask_wgh;
  // [Note: Per-channel quantization mask setting]
  // Per-channel quantization is on weight output channel mostly, mask_wgh= 1
  // here means 2^0. 0 means the 0th dimension of wgh tensor, aka output
  // channel. DNN requires mask = 2^k for the kth axis to be quantized. Only one
  // axis quantization is supported in IPEX. Multi channel quantization is not
  // supported. In addition, src, dst should still be per-tensor quant, aka
  // mask=0. Per-channel quantization on activation is not supported in conv.
  mask_wgh = (wgh.qscheme() == kPerTensorAffine) ? 0 : 1;
  primitive_attr pattr;

#ifdef USE_PRIMITIVE_CACHE
  create_key(
      key_primitive,
      src_dims,
      wgh_dims,
      src_data_t,
      dst_data_t,
      groups,
      _stride,
      _dilation,
      _padding_front_top_left,
      _padding_back_bottom_right,
      is_onednn_layout_suggested,
      is_channels_last_suggested,
      attr);
#endif

  convolution_forward conv_forward;
  convolution_forward::primitive_desc conv_fwd_pd;

#ifdef USE_PRIMITIVE_CACHE
  bool load_from_cache = find_key<convolution_forward>(key_primitive);
#else
  bool load_from_cache = false;
#endif

// [Note: Use symmetric quant implementation when zp is 0]
// (JIRA: https://jira.devtools.intel.com/browse/MFDNN-9633)
// Due to asymmetric quant has perf gap compared to symm quant, we need to avoid
// dnn kernel goes into asymm path if tensor zp is 0. We expect following
// behaviour:
// 1. IF IPEX is Symmetric only: Alwasy refuse to use runtime zp. Use symmetric
// kernel.
// 2. IF IPEX is Asymmetric supported:
//      a. Check src&dzp&wgh zp, if all are zero, we go into symmetric path for
//      perf. With this WA, operate like conv_relu fusion would maintin high
//      perf even the overall config is asymm.
//      b. If zp is not zero, using asymmetric kernel. Perf regression should
//      then happen.
#ifdef BUILD_PRIOR_SYMM_QUANT
  bool src_need_zp = requires_runtime_zp(src);
  bool dst_need_zp = requires_runtime_zp(dst);
  bool wgh_need_zp = requires_runtime_zp(wgh);
#endif

  std::tie(src_usr_md, wgh_usr_md, dst_usr_md) =
      qconv_get_usr_md(src, wgh, dst, groups, memory_layout_for_conv);

  if (load_from_cache) {
    conv_forward = fetch_m<convolution_forward>(key_primitive);
    auto conv_fwd_pd_t = conv_forward.get_primitive_desc();
    conv_fwd_pd = convolution_forward::primitive_desc(
        const_cast<dnnl_primitive_desc_t>(conv_fwd_pd_t));
  } else {
    if (is_onednn_layout_suggested) {
      std::tie(src_md, wgh_md, dst_md) =
          qconv_get_blocked_md(src, src_usr_md, wgh_usr_md, dst_usr_md);
    } else {
      auto ic = src.size(1);
      auto oc = dst.size(1);
      memory::dims wgh_tz =
          compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());
      std::tie(src_md, wgh_md, dst_md) = qconv_get_plain_md(
          src_usr_md,
          wgh_usr_md,
          dst_usr_md,
          wgh_tz,
          is_channels_last_suggested);
    }

    pattr.set_scales_mask(DNNL_ARG_SRC, mask_ac);
    pattr.set_scales_mask(DNNL_ARG_WEIGHTS, mask_wgh);
    pattr.set_post_ops(po);

#ifdef BUILD_PRIOR_SYMM_QUANT
    // Only setting zp mask when zp is not zero
    // See: [Note: Use symmetric quant implementation when zp is 0]
    if (src_need_zp)
      pattr.set_zero_points_mask(DNNL_ARG_DST, mask_ac);
    if (dst_need_zp)
      pattr.set_zero_points_mask(DNNL_ARG_SRC, mask_ac);
    if (wgh_need_zp)
      pattr.set_zero_points_mask(DNNL_ARG_WEIGHTS, mask_wgh);
#endif
#ifdef USE_SCRATCHPAD_MODE
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

    // create primitive
    conv_fwd_pd = convolution_forward::primitive_desc(
        engine,
        prop_kind::forward,
        algorithm::convolution_direct,
        src_md,
        wgh_md,
        memory::desc(),
        dst_md,
        _stride,
        _dilation,
        _padding_front_top_left,
        _padding_back_bottom_right,
        pattr);

#ifdef USE_PRIMITIVE_CACHE
    conv_forward =
        create_and_fetch_m<convolution_forward>(key_primitive, conv_fwd_pd);
#else
    conv_forward = convolution_forward(conv_fwd_pd);
#endif
  }

  auto weight_cache_optimization = [&]() {
    // TODO:: remove ChannelsLast option after oneDNN fix accuracy issue
    return (memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked ||
            memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::ChannelsLast) &&
        !at::GradMode::is_enabled();
  }();

  memory src_m, wgh_m, dst_m;
  Tensor src_blocked, wgh_blocked, dst_blocked = dst;
  if (is_onednn_layout_suggested) {
    auto expected_src_md = conv_fwd_pd.src_desc();
    auto expected_wgh_md = conv_fwd_pd.weights_desc();
    auto expected_dst_md = conv_fwd_pd.dst_desc();
    src_m = qconv_get_expected_src_memory(
        src, src_blocked, src_usr_md, expected_src_md, engine, load_from_cache);
    wgh_m = qconv_get_expected_wgh_memory(
        wgh,
        wgh_blocked,
        wgh_usr_md,
        expected_wgh_md,
        wgh_scales,
        engine,
        weight_cache_optimization);
    dst_m = qconv_get_blocked_dst_memory(
        dst, dst_blocked, dst_usr_md, expected_dst_md, attr, engine);
  } else {
    src_m = dpcpp_onednn_memory(src_usr_md, engine, src.data_ptr());
    dst_m = dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr());
    wgh_m = dpcpp_onednn_memory(wgh_usr_md, engine, wgh.data_ptr());
    if (memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::ChannelsLast) {
      // TODO: Should remove after oneDNN fix the accuracy issue
      auto expected_wgh_md = conv_fwd_pd.weights_desc();
      wgh_m = qconv_get_expected_wgh_memory(
          wgh,
          wgh_blocked,
          wgh_usr_md,
          expected_wgh_md,
          wgh_scales,
          engine,
          weight_cache_optimization);
    }
  }

  std::unordered_map<int, memory> args;
  if (attr.with_binary())
    attr.construct_post_binary(conv_fwd_pd, po, args);
  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_WEIGHTS, wgh_m});
  args.insert({DNNL_ARG_DST, dst_m});

  memory src_sc_m, src_zp_m;
  std::tie(src_sc_m, src_zp_m) = q_get_sc_zp_gpu_mem(src, engine);
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_sc_m});

#ifdef BUILD_PRIOR_SYMM_QUANT
  // Only setting zp when zp is not zero
  // See: [Note: Use symmetric quant implementation when zp is 0]
  Tensor srz_zp;
  memory::desc src_zp_md;
  if (src_need_zp) {
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zp_m});
  }
#endif

  // dst scale is no need for setting, since it is fused in postop via linear

#ifdef BUILD_PRIOR_SYMM_QUANT
  // Only setting zp when zp is not zero
  // See: [Note: Use symmetric quant implementation when zp is 0]
  memory dst_zp_m;
  if (dst_need_zp) {
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zp_m});
  }
#endif

#ifdef USE_SCRATCHPAD_MODE
  size_t scratchpad_size = conv_fwd_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, src.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = dpcpp_onednn_memory(
      conv_fwd_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});
#endif

  if (wgh.qscheme() == kPerTensorAffine) {
    memory wgh_sc_m;
    wgh_sc_m = q_get_wgh_sc_gpu_mem(wgh, engine);
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wgh_sc_m});
    DPCPP_ONEDNN_EXEC(conv_forward, strm, args);
  } else {
    // Per-channel quantized
    Tensor wgh_sc = wgh.q_per_channel_scales();
    memory::desc wgh_sc_md = memory::desc(
        get_onednn_dims(wgh_sc), memory::data_type::f32, memory::format_tag::x);
    memory wgh_sc_m = dpcpp_onednn_memory(wgh_sc_md, engine, wgh_sc.data_ptr());
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wgh_sc_m});
    DPCPP_ONEDNN_EXEC(conv_forward, strm, args);
  }

  if (is_onednn_layout_suggested && dst_blocked.data_ptr() != dst.data_ptr()) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(dst_blocked);
    DPCPPTensorContext::set_tensor_ctx(dst, std::move(blk_ctx));
  }

  return dst;
}

} // namespace oneDNN
} // namespace xpu
