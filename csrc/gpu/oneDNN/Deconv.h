#pragma once

#include <ATen/ATen.h>
#include <oneDNN/Runtime.h>
#include <oneDNN/Utils.h>
#include <runtime/Utils.h>
#include <tensor/Context.h>
#include <utils/LRUCache.h>

#include <oneapi/dnnl/dnnl.hpp>
#include "Attr.h"

using namespace xpu::dpcpp;

namespace xpu {
namespace oneDNN {
namespace {

static inline memory::dims deconv_compatible_dilation(IntArrayRef& dilation) {
  memory::dims ret = dilation.vec();
  for (auto it = ret.begin(); it != ret.end(); it++) {
    *it -= 1;
  }
  return ret;
}

static inline memory::dims deconv_dst_tz(
    IntArrayRef src_size,
    IntArrayRef wgh_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    IntArrayRef dst_padding,
    int64_t groups) {
  auto dim = src_size.size();
  memory::dims dst_size(dim);
  auto kernel_size = wgh_size.slice(2);

  dst_size[0] = src_size[0];
  dst_size[1] = wgh_size[1] * groups;
  for (size_t d = 2; d < dim; ++d) {
    dst_size[d] = (src_size[d] - 1) * stride[d - 2] - 2 * padding[d - 2] +
        (dilation[d - 2] * (kernel_size[d - 2] - 1) + 1) + dst_padding[d - 2];
  }
  return dst_size;
}

static inline memory::format_tag deconv_src_fmt(
    const int64_t ndim,
    const bool is_channels_last = false) {
  // 3D: n/c/w (n/w/c)         [a/b/c (a/c/b)]
  // 4D: n/c/h/w (n/h/w/c)     [a/b/c/d (a/c/d/b)]
  // 5D: n/c/d/h/w (n/d/h/w/c) [a/b/c/d/e (a/c/d/e/b)]
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

static inline std::vector<int64_t> deconv_wgh_fmt(
    const Tensor& wgh,
    const int64_t ndim,
    memory::dims wgh_tz,
    const bool grouped = false,
    const bool is_channels_last = false) {
  // 3D fmt: (g)i/o/w ((g)i/w/o)  [b/a/c  (b/c/a)]
  // 4D fmt: (g)i/o/h/w ((g)i/h/w/o) [b/a/c/d (b/c/d/a)]
  // 5D fmt: (g)i/o/d/h/w ((g)i/d/h/w/o) [b/a/c/d/e (b/c/d/e/a)]
  auto strides_ = wgh.strides().vec();
  std::vector<int64_t> strides;
  if (grouped) {
    strides = compatible_groups_deconv_strides(wgh, wgh_tz);
  } else {
    strides = strides_;
    std::swap(strides[0], strides[1]);
  }
  return strides;
}

static inline memory::dims deconv_compatible_wgh_dims(
    int64_t ndim,
    int64_t groups,
    int64_t oc,
    int64_t ic,
    IntArrayRef wgh_tz) {
  if (ndim == 3) {
    auto kw = wgh_tz[2];
    return (groups != 1) ? memory::dims({groups, oc / groups, ic / groups, kw})
                         : memory::dims({oc, ic, kw});
  } else if (ndim == 4) {
    auto kh = wgh_tz[2];
    auto kw = wgh_tz[3];
    return (groups != 1)
        ? memory::dims({groups, oc / groups, ic / groups, kh, kw})
        : memory::dims({oc, ic, kh, kw});
  } else if (ndim == 5) {
    auto kd = wgh_tz[2];
    auto kh = wgh_tz[3];
    auto kw = wgh_tz[4];
    return (groups != 1)
        ? memory::dims({groups, oc / groups, ic / groups, kd, kh, kw})
        : memory::dims({oc, ic, kd, kh, kw});
  } else {
    TORCH_CHECK(0, "unsupported dimension in xpu oneDNN deconvolution...");
  }
}
} // namespace

static std::tuple<
    memory::desc,
    memory::desc,
    memory::desc,
    memory::desc,
    memory::desc,
    memory::desc>
deconv_get_plain_md(
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& dst,
    int64_t groups,
    bool is_channels_last_suggested) {
  auto ndim = src.ndimension();
  auto src_data_t = get_onednn_dtype_include_double(src);
  auto fmt_src = deconv_src_fmt(ndim, is_channels_last_suggested);
  auto src_usr_md = memory::desc(src.sizes().vec(), src_data_t, fmt_src);

  auto dst_data_t = get_onednn_dtype_include_double(dst);
  auto dst_usr_md = memory::desc(dst.sizes().vec(), dst_data_t, fmt_src);

  auto ic = src.size(1);
  auto oc = dst.size(1);
  memory::dims wgh_tz =
      deconv_compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());
  auto wgh_dt = get_onednn_dtype_include_double(wgh);
  auto fmt_wgh = deconv_wgh_fmt(
      wgh, ndim, wgh_tz, groups != 1, is_channels_last_suggested);
  memory::desc wgh_usr_md = memory::desc(wgh_tz, wgh_dt, fmt_wgh);

  return {
      src_usr_md, wgh_usr_md, dst_usr_md, src_usr_md, wgh_usr_md, dst_usr_md};
}

static std::tuple<
    memory::desc,
    memory::desc,
    memory::desc,
    memory::desc,
    memory::desc,
    memory::desc>
deconv_get_blocked_md(
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& dst,
    int64_t groups) {
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md;
  auto ndim = src.ndimension();
  auto fmt_src = deconv_src_fmt(ndim);
  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  if (src_ctx.is_plain()) {
    auto src_data_t = get_onednn_dtype_include_double(src);
    src_usr_md = memory::desc(src.sizes().vec(), src_data_t, fmt_src);
  } else {
    src_usr_md = src_ctx.meta();
  }

  auto dst_ctx = DPCPPTensorContext::get_tensor_ctx(dst);
  if (dst_ctx.is_plain()) {
    auto dst_data_t = get_onednn_dtype_include_double(dst);
    dst_usr_md = memory::desc(dst.sizes().vec(), dst_data_t, fmt_src);
  } else {
    dst_usr_md = dst_ctx.meta();
  }

  auto wgh_ctx = DPCPPTensorContext::get_tensor_ctx(wgh);
  if (wgh_ctx.is_plain()) {
    auto ic = src.size(1);
    auto oc = dst.size(1);
    memory::dims wgh_tz =
        deconv_compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());
    auto wgh_dt = get_onednn_dtype_include_double(wgh);
    auto fmt_wgh = deconv_wgh_fmt(wgh, ndim, wgh_tz, groups != 1);
    wgh_usr_md = memory::desc(wgh_tz, wgh_dt, fmt_wgh);
  } else {
    wgh_usr_md = wgh_ctx.meta();
  }

  // create memory desc for deconv primitive and query the blocked format
  auto fmt_any = memory::format_tag::any;
  auto src_md =
      memory::desc(src_usr_md.get_dims(), src_usr_md.get_data_type(), fmt_any);
  auto wgh_md =
      memory::desc(wgh_usr_md.get_dims(), wgh_usr_md.get_data_type(), fmt_any);
  auto dst_md =
      memory::desc(dst_usr_md.get_dims(), dst_usr_md.get_data_type(), fmt_any);

  return {src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md};
}

static memory deconv_get_expected_src_memory(
    const at::Tensor& src,
    at::Tensor& src_blocked,
    memory::desc& src_usr_md,
    memory::desc& expected_src_md,
    dnnl::engine& engine,
    bool need_reorder = true) {
  memory src_m;
  if (src_usr_md != expected_src_md) {
    // avoid reorder in case of, [n][C][1][1][16c] <==> [n][c][1][1]
    if (src.sizes().size() == 4 && src.size(2) == 1 && src.size(3) == 1) {
      src_m = dpcpp_onednn_memory(expected_src_md, engine, src.data_ptr());
    } else {
      src_blocked =
          empty_opaque_tensor(expected_src_md, src.options(), c10::nullopt);
      src_m =
          dpcpp_onednn_memory(expected_src_md, engine, src_blocked.data_ptr());
      if (need_reorder)
        xpu::oneDNN::reorder(src, src_blocked);
    }
  } else {
    src_m = dpcpp_onednn_memory(src_usr_md, engine, src.data_ptr());
    src_blocked = src;
  }
  return src_m;
}

static memory deconv_get_expected_wgh_memory(
    const at::Tensor& wgh,
    at::Tensor& wgh_blocked,
    memory::desc& wgh_usr_md,
    memory::desc& expected_wgh_md,
    dnnl::engine& engine,
    bool weight_cache_optimization,
    bool need_reorder = true) {
  memory wgh_m;
  if (wgh_usr_md != expected_wgh_md) {
    wgh_blocked =
        empty_opaque_tensor(expected_wgh_md, wgh.options(), c10::nullopt);
    wgh_m =
        dpcpp_onednn_memory(expected_wgh_md, engine, wgh_blocked.data_ptr());

    if (need_reorder) {
      // reshape for group convolution weight
      Tensor reshaped_wgh;
      if (wgh_blocked.ndimension() > wgh.ndimension()) {
        // for groups conv case:
        // expected_wgh will be 5-D Tensor based on expected_wgh_md:
        // g/o/i/h/w or g/o/h/w/i
        // wgh will be 4-D Tensor based on PyTorch
        // (g)i/o/h/w or (g)o/h/w/i
        // we need to manually reshape 4-D wgh to 5-D,
        // consistent with expected_wgh
        reshaped_wgh = share_storage_and_set_strided_as(
            wgh,
            wgh_blocked.sizes(),
            /*compatible with different strides of weight (including contiguous,
               channels_last and non-contiguous) */
            compatible_groups_deconv_strides(wgh, wgh_blocked.sizes().vec()),
            c10::nullopt);
      } else {
        // PyTorch deconv weight dim g/i/o/h/w or g/i/h/w/o
        // oneDNN deconv weight dim g/o/i/h/w or g/o/h/w/i
        reshaped_wgh = wgh.transpose(0, 1);
      }
      xpu::oneDNN::reorder(reshaped_wgh, wgh_blocked);

      if (weight_cache_optimization) {
        auto wgh_opt_ctx = DPCPPTensorContext::release_tensor_ctx(wgh_blocked);
        wgh_opt_ctx.set_aten_meta(
            {reshaped_wgh.sizes().vec(), reshaped_wgh.strides().vec()});
        DPCPPTensorContext::set_tensor_ctx(wgh, std::move(wgh_opt_ctx));
      }
    }
  } else {
    wgh_m = dpcpp_onednn_memory(wgh_usr_md, engine, wgh.data_ptr());
    wgh_blocked = wgh;
  }
  return wgh_m;
}

static memory deconv_get_expected_dst_memory(
    const at::Tensor& dst,
    at::Tensor& dst_blocked,
    memory::desc& dst_usr_md,
    memory::desc& expected_dst_md,
    dnnl::engine& engine,
    bool need_reorder = true) {
  memory dst_m;
  if (dst_usr_md != expected_dst_md) {
    dst_blocked =
        empty_opaque_tensor(expected_dst_md, dst.options(), c10::nullopt);
    dst_m =
        dpcpp_onednn_memory(expected_dst_md, engine, dst_blocked.data_ptr());
    if (need_reorder)
      xpu::oneDNN::reorder(dst, dst_blocked);
  } else {
    dst_m = dpcpp_onednn_memory(dst_usr_md, engine, dst.data_ptr());
    dst_blocked = dst;
  }
  return dst_m;
}

static void deconvolution(
    Tensor& dst,
    const Tensor& src,
    const Tensor& wgh,
    const Tensor& bia,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dst_padding,
    IntArrayRef dilation,
    int64_t groups,
    Attr& attr) {
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto ndim = src.ndimension();
  auto memory_layout_for_conv = get_memory_layout_for_conv(src, wgh);
  bool is_onednn_layout_suggested =
      memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked;

  // create usr_md for tensors, and md for conv primitive
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md;
  if (is_onednn_layout_suggested) {
    std::tie(src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md) =
        deconv_get_blocked_md(src, wgh, dst, groups);
  } else {
    bool is_channels_last_suggested =
        memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::ChannelsLast;
    std::tie(src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md) =
        deconv_get_plain_md(src, wgh, dst, groups, is_channels_last_suggested);
  }
  memory::format_tag bia_fmt = memory::format_tag::x;
  auto bia_md = bia.defined()
      ? memory::desc(
            {dst.size(1)}, get_onednn_dtype_include_double(bia), bia_fmt)
      : memory::desc();

  // create primitive desc
  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = deconv_compatible_dilation(dilation);

  // construct primitive attr
  primitive_attr pattr;
  post_ops po;
  attr.extract_post_ops(po, dst);
  pattr.set_post_ops(po);

#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

  if (src_usr_md.get_data_type() == memory::data_type::f32) {
    pattr.set_fpmath_mode(xpu::oneDNN::get_onednn_fpmath_mode());
  }

  auto deconv_fwd_pd = deconvolution_forward::primitive_desc(
      engine,
      prop_kind::forward,
      algorithm::deconvolution_direct,
      src_md,
      wgh_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding,
      _padding,
      pattr);

  auto weight_cache_optimization = [&]() {
    bool onoff = false;
    onoff |= is_onednn_layout_suggested;
    onoff &= !at::GradMode::is_enabled();
    return onoff;
  }();

  memory src_m, wgh_m, dst_m, bia_m;
  Tensor src_blocked, wgh_blocked, dst_blocked = dst;
  if (is_onednn_layout_suggested) {
    auto expected_src_md = deconv_fwd_pd.src_desc();
    auto expected_wgh_md = deconv_fwd_pd.weights_desc();
    auto expected_dst_md = deconv_fwd_pd.dst_desc();
    src_m = deconv_get_expected_src_memory(
        src, src_blocked, src_usr_md, expected_src_md, engine);
    wgh_m = deconv_get_expected_wgh_memory(
        wgh,
        wgh_blocked,
        wgh_usr_md,
        expected_wgh_md,
        engine,
        weight_cache_optimization);
    dst_m = deconv_get_expected_dst_memory(
        dst, dst_blocked, dst_usr_md, expected_dst_md, engine, attr.with_sum());
  } else {
    src_m = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
    wgh_m = dpcpp_onednn_memory(wgh_md, engine, wgh.data_ptr());
    dst_m = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());
  }

  std::unordered_map<int, memory> args;
  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_WEIGHTS, wgh_m});
  args.insert({DNNL_ARG_DST, dst_m});

  if (bia.defined()) {
    auto bia_m = dpcpp_onednn_memory(bia_md, engine, bia.data_ptr());
    args.insert({DNNL_ARG_BIAS, bia_m});
  }
  if (attr.with_binary())
    attr.construct_post_binary(deconv_fwd_pd, po, args);

#ifdef USE_SCRATCHPAD_MODE
  size_t scratchpad_size = deconv_fwd_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, src.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = dpcpp_onednn_memory(
      deconv_fwd_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});
#endif

  auto deconv_fwd = deconvolution_forward(deconv_fwd_pd);
  DPCPP_ONEDNN_EXEC(deconv_fwd, strm, args);

  // propagate blk format from input to output
  if (is_onednn_layout_suggested && dst_blocked.data_ptr() != dst.data_ptr()) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(dst_blocked);
    DPCPPTensorContext::set_tensor_ctx(dst, std::move(blk_ctx));
  }
}

static void deconvolution_backward_data(
    at::Tensor& diff_src,
    const Tensor& diff_dst,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined) {
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto memory_layout_for_conv = get_memory_layout_for_conv(diff_dst, weight);
  bool is_onednn_layout_suggested =
      memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked;
  // create memory desc
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md;
  if (is_onednn_layout_suggested) {
    std::tie(src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md) =
        deconv_get_blocked_md(diff_src, weight, diff_dst, groups);
  } else {
    bool is_channels_last_suggested =
        memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::ChannelsLast;
    std::tie(src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md) =
        deconv_get_plain_md(
            diff_src, weight, diff_dst, groups, is_channels_last_suggested);
  }
  memory::format_tag bia_fmt = memory::format_tag::x;
  auto bia_md = bias_defined
      ? memory::desc({diff_dst.size(1)}, wgh_md.get_data_type(), bia_fmt)
      : memory::desc();

  // create fwd primitive desc hint
  primitive_attr pattr;
  if (dst_usr_md.get_data_type() == memory::data_type::f32) {
    pattr.set_fpmath_mode(xpu::oneDNN::get_onednn_fpmath_mode());
  }
#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = deconv_compatible_dilation(dilation);
  auto deconv_fwd_pd = deconvolution_forward::primitive_desc(
      engine,
      prop_kind::forward,
      algorithm::deconvolution_direct,
      src_md,
      wgh_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding,
      _padding,
      pattr);

  // create bwd primitive desc
  auto deconv_backward_data_pd = deconvolution_backward_data::primitive_desc(
      engine,
      algorithm::deconvolution_direct,
      src_md,
      wgh_md,
      dst_md,
      _stride,
      _dilation,
      _padding,
      _padding,
      deconv_fwd_pd);

  // create memory
  Tensor expected_src, expected_wei, expected_dst;
  memory diff_dst_m, wei_m, diff_src_m;
  if (is_onednn_layout_suggested) {
    auto expected_src_md = deconv_backward_data_pd.diff_src_desc();
    auto expected_wgh_md = deconv_backward_data_pd.weights_desc();
    auto expected_dst_md = deconv_backward_data_pd.diff_dst_desc();
    diff_src_m = deconv_get_expected_src_memory(
        diff_src,
        expected_src,
        src_usr_md,
        expected_src_md,
        engine,
        false /* need_reorder*/);
    wei_m = deconv_get_expected_wgh_memory(
        weight,
        expected_wei,
        wgh_usr_md,
        expected_wgh_md,
        engine,
        false /* weight_cache */);
    diff_dst_m = deconv_get_expected_dst_memory(
        diff_dst, expected_dst, dst_usr_md, expected_dst_md, engine);
  } else {
    diff_src_m = dpcpp_onednn_memory(src_usr_md, engine, diff_src.data_ptr());
    wei_m = dpcpp_onednn_memory(wgh_usr_md, engine, weight.data_ptr());
    diff_dst_m = dpcpp_onednn_memory(dst_usr_md, engine, diff_dst.data_ptr());
  }

  // insert args
  std::unordered_map<int, memory> args;
#ifdef USE_SCRATCHPAD_MODE
  size_t scratchpad_size = deconv_backward_data_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, diff_dst.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_memory = dpcpp_onednn_memory(
      deconv_backward_data_pd.scratchpad_desc(),
      engine,
      scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});
#endif
  args.insert({DNNL_ARG_DIFF_DST, diff_dst_m});
  args.insert({DNNL_ARG_WEIGHTS, wei_m});
  args.insert({DNNL_ARG_DIFF_SRC, diff_src_m});

  // execute primitive
  auto deconv_backward_data =
      dnnl::deconvolution_backward_data(deconv_backward_data_pd);
  DPCPP_ONEDNN_EXEC(deconv_backward_data, strm, args);

  // propagate blk from grad_output to grad_input
  if (is_onednn_layout_suggested &&
      diff_src.data_ptr() != expected_src.data_ptr()) {
    auto blk_ctx = DPCPPTensorContext::release_tensor_ctx(expected_src);
    DPCPPTensorContext::set_tensor_ctx(diff_src, std::move(blk_ctx));
  }
}

static void deconvolution_backward_weights(
    at::Tensor& diff_wgh,
    at::Tensor& diff_bia,
    const Tensor& diff_dst,
    const Tensor& src,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto memory_layout_for_conv = get_memory_layout_for_conv(src, diff_dst);
  bool is_onednn_layout_suggested =
      memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::Blocked;

  // create memory desc
  memory::desc src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md;
  if (is_onednn_layout_suggested) {
    std::tie(src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md) =
        deconv_get_blocked_md(src, diff_wgh, diff_dst, groups);
  } else {
    bool is_channels_last_suggested =
        memory_layout_for_conv == MEMORY_LAYOUT_FOR_CONV::ChannelsLast;
    std::tie(src_usr_md, wgh_usr_md, dst_usr_md, src_md, wgh_md, dst_md) =
        deconv_get_plain_md(
            src, diff_wgh, diff_dst, groups, is_channels_last_suggested);
  }
  memory::format_tag bia_fmt = memory::format_tag::x;
  auto bia_md = diff_bia.defined()
      ? memory::desc({diff_dst.size(1)}, src_md.get_data_type(), bia_fmt)
      : memory::desc();

  // create fwd primitive desc hint
  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = deconv_compatible_dilation(dilation);
  primitive_attr pattr;
  if (src_usr_md.get_data_type() == memory::data_type::f32) {
    pattr.set_fpmath_mode(xpu::oneDNN::get_onednn_fpmath_mode());
  }
#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif
  auto deconv_fwd_pd = deconvolution_forward::primitive_desc(
      engine,
      prop_kind::forward,
      algorithm::deconvolution_direct,
      src_md,
      wgh_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding,
      _padding,
      pattr);

  auto deconv_bwd_w_pd = deconvolution_backward_weights::primitive_desc(
      engine,
      algorithm::deconvolution_direct,
      src_md,
      wgh_md,
      bia_md,
      dst_md,
      _stride,
      _dilation,
      _padding,
      _padding,
      deconv_fwd_pd,
      pattr);

  // create bwd memory
  Tensor expected_src, expected_diff_dst, expected_diff_wgh;
  memory src_m, diff_dst_m, diff_wgh_m;
  if (is_onednn_layout_suggested) {
    auto expected_src_md = deconv_bwd_w_pd.src_desc();
    auto expected_dst_md = deconv_bwd_w_pd.diff_dst_desc();
    auto expected_wgh_md = deconv_bwd_w_pd.diff_weights_desc();
    src_m = deconv_get_expected_src_memory(
        src, expected_src, src_usr_md, expected_src_md, engine);
    diff_wgh_m = deconv_get_expected_wgh_memory(
        diff_wgh,
        expected_diff_wgh,
        wgh_usr_md,
        expected_wgh_md,
        engine,
        false /* weight_cache */,
        false /* need_reorder */);
    diff_dst_m = deconv_get_expected_dst_memory(
        diff_dst, expected_diff_dst, dst_usr_md, expected_dst_md, engine);
  } else {
    src_m = dpcpp_onednn_memory(src_usr_md, engine, src.data_ptr());
    diff_dst_m = dpcpp_onednn_memory(dst_usr_md, engine, diff_dst.data_ptr());
    diff_wgh_m = dpcpp_onednn_memory(wgh_usr_md, engine, diff_wgh.data_ptr());
  }

  // insert args
  std::unordered_map<int, memory> args;
  args.insert({DNNL_ARG_DIFF_DST, diff_dst_m});
  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_DIFF_WEIGHTS, diff_wgh_m});

  if (diff_bia.defined()) {
    memory diff_bia_m =
        dpcpp_onednn_memory(bia_md, engine, diff_bia.data_ptr());
    args.insert({DNNL_ARG_DIFF_BIAS, diff_bia_m});
  }

#ifdef USE_SCRATCHPAD_MODE
  int scratchpad_size = deconv_bwd_w_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, src.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = dpcpp_onednn_memory(
      deconv_bwd_w_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});
#endif

  // execute primitive
  auto deconv_bwd_w = dnnl::deconvolution_backward_weights(deconv_bwd_w_pd);
  DPCPP_ONEDNN_EXEC(deconv_bwd_w, strm, args);

  // reorder weight to algin PyTorch format if necessary
  if (diff_wgh_m.get_desc() != wgh_usr_md) {
    Tensor reshaped_diff_wgh;
    if (expected_diff_wgh.ndimension() > diff_wgh.ndimension()) {
      reshaped_diff_wgh = share_storage_and_set_strided_as(
          diff_wgh,
          expected_diff_wgh.sizes(),
          compatible_groups_deconv_strides(
              diff_wgh, expected_diff_wgh.sizes().vec()),
          c10::nullopt);
    } else {
      reshaped_diff_wgh = diff_wgh.transpose(0, 1);
    }
    xpu::oneDNN::reorder(expected_diff_wgh, reshaped_diff_wgh);
  }
}

} // namespace oneDNN
} // namespace xpu
