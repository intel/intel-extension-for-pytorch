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

static std::tuple<memory::desc, memory::desc, memory::desc>
qconv_get_plain_usr_md(
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& dst,
    int64_t groups,
    bool is_channels_last_suggested) {
  auto ndim = src.ndimension();
  auto src_data_t =
      is_opaque_u8(src) ? memory::data_type::s8 : get_onednn_dtype(src);
  // 3D: n/c/w (n/w/c)
  // 4D: n/c/h/w (n/h/w/c)
  // 5D: n/c/d/h/w (n/d/h/w/c)
  auto fmt_src = conv_src_fmt(ndim, is_channels_last_suggested);
  auto src_usr_md = memory::desc(src.sizes().vec(), src_data_t, fmt_src);

  auto dst_data_t = get_onednn_dtype(dst);
  auto dst_usr_md = memory::desc(dst.sizes().vec(), dst_data_t, fmt_src);

  memory::desc wgh_usr_md, wgh_md;
  auto ic = src.size(1);
  auto oc = dst.size(1);
  memory::dims wgh_tz = compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());
  auto wei_data_t = memory::data_type::s8;
  // 3D: (g)o/i/w ((g)o/w/i)
  // 4D: (g)o/i/h/w ((g)o/h/w/i)
  // 5D: (g)o/i/d/h/w ((g)o/d/h/w/i)
  auto fmt_wgh = conv_wgh_fmt(ndim, groups != 1, is_channels_last_suggested);
  if (is_channels_last_suggested) {
    // TODO: remove this path when oneDNN fix the accuracy issue.
    // in ChannelsLast senario, fmt_wgh should be nhwc instead of any
    wgh_usr_md = memory::desc(wgh_tz, wei_data_t, fmt_wgh);
  } else {
    wgh_usr_md = memory::desc(wgh_tz, wei_data_t, fmt_wgh);
  }

  return {src_usr_md, wgh_usr_md, dst_usr_md};
}

static std::tuple<memory::desc, memory::desc, memory::desc>
qconv_get_blocked_usr_md(
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& dst,
    int64_t groups) {
  // create memory desc from the src/wgh/dst tensors
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md;
  auto ndim = src.ndimension();
  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  auto fmt_src = conv_src_fmt(ndim);

  if (src_ctx.is_plain()) {
    auto src_tz = src.sizes().vec();
    auto src_data_t =
        is_opaque_u8(src) ? memory::data_type::s8 : get_onednn_dtype(src);
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
    auto fmt_wgh = conv_wgh_fmt(ndim, groups != 1);
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
      : memory::desc(src_usr_md.dims(), src_usr_md.data_type(), fmt_any);
  wgh_md = memory::desc(wgh_usr_md.dims(), memory::data_type::s8, fmt_any);
  dst_md = memory::desc(dst_usr_md.dims(), dst_usr_md.data_type(), fmt_any);

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

static inline void get_conv_scale(
    const Tensor& src,
    const Tensor& wgh,
    const memory::data_type& src_data_t,
    int oc,
    std::vector<float>& wgh_scales,
    std::vector<float>& conv_scale,
    int& mask_conv) {
  float src_scale;
  if (wgh.qscheme() == kPerTensorAffine) {
    wgh_scales.push_back(static_cast<float>(wgh.q_scale()));
  } else {
    for (int i = 0; i < oc; i++) {
      wgh_scales.push_back(wgh.q_per_channel_scales()[i].item<float>());
    }
  }

  // TODO: scale setting in separate functions
  src_scale = (src_data_t == memory::data_type::u8 && src.q_zero_point() == 128)
      ? src.q_scale() / 2
      : src.q_scale();
  conv_scale.clear();
  /* Note: [Convolution requantization]
      Suppose y = w * x. The * refer to convolution operation, and y, w, x are
      dtype of FP32.
      Then we have
        y_int / y_sc =  (w_int / w_sc) * (x_int / x_sc) =>
        y_int = [y_sc / (w_sc * x_sc)] (w_int * x_int).
      The y_sc / (w_sc x x_sc) is requantization scale, which is also the
      conv_scale in following line.
      Inversion is required due to scale_onednn = 1  / scale_torch */
  /*The requantization will be performed in Attr.h with appending post op
   * linear to adjust the scale/zeropoint*/
  for (int i = 0; i < wgh_scales.size(); i++) {
    conv_scale.push_back(
        src_scale * wgh_scales[i]); // 1.f / (1.f / (src_scale * wgh_scales[i]))
  }
  mask_conv = wgh_scales.size() > 1 ? 1 << 1 : 0;
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
  auto src_data_t = is_opaque_u8(src) ? memory::data_type::s8
                                      : get_onednn_dtype_include_double(src);
  auto dst_data_t = get_onednn_dtype_include_double(dst);
  // conv config
  memory::dims _stride = stride.vec();
  memory::dims _padding_front_top_left = padding_front_top_left.vec();
  memory::dims _padding_back_bottom_right = padding_back_bottom_right.vec();
  memory::dims _dilation = compatible_dilation(dilation);
  // conv post ops config
  post_ops po;
  attr.extract_post_ops(po, dst);
  // conv scale and zero_point
  std::vector<float> wgh_scales, conv_scale = {1};
  int conv_zero_point = 0, mask_ac = 0, mask_conv;
  get_conv_scale(
      src, wgh, src_data_t, dst.size(1), wgh_scales, conv_scale, mask_conv);

  lru_key_t key_primitive;
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
      attr,
      conv_scale,
      conv_zero_point);
#endif

  convolution_forward conv_forward;
  convolution_forward::primitive_desc conv_fwd_pd;

#ifdef USE_PRIMITIVE_CACHE
  bool load_from_cache = find_key<convolution_forward>(key_primitive);
#else
  bool load_from_cache = false;
#endif

  if (is_onednn_layout_suggested) {
    std::tie(src_usr_md, wgh_usr_md, dst_usr_md) =
        qconv_get_blocked_usr_md(src, wgh, dst, groups);
  } else {
    std::tie(src_usr_md, wgh_usr_md, dst_usr_md) = qconv_get_plain_usr_md(
        src, wgh, dst, groups, is_channels_last_suggested);
  }

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

    auto conv_fwd_desc = convolution_forward::desc(
        prop_kind::forward,
        algorithm::convolution_direct,
        src_md,
        wgh_md,
        memory::desc(),
        dst_md,
        _stride,
        _dilation,
        _padding_front_top_left,
        _padding_back_bottom_right);

    primitive_attr pattr;
    pattr.set_output_scales(mask_conv, conv_scale);
    pattr.set_zero_points(DNNL_ARG_DST, mask_ac, {conv_zero_point});
    pattr.set_post_ops(po);

#ifdef USE_SCRATCHPAD_MODE
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

    // create primitive
    conv_fwd_pd =
        convolution_forward::primitive_desc(conv_fwd_desc, pattr, engine);

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

#ifdef USE_SCRATCHPAD_MODE
  int scratchpad_size = conv_fwd_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, src.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = dpcpp_onednn_memory(
      conv_fwd_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});
#endif

  DPCPP_ONEDNN_EXEC(conv_forward, strm, args);
  if (is_onednn_layout_suggested && dst_blocked.data_ptr() != dst.data_ptr()) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(dst_blocked);
    DPCPPTensorContext::set_tensor_ctx(dst, std::move(blk_ctx));
  }

  return dst;
}

} // namespace oneDNN
} // namespace xpu
