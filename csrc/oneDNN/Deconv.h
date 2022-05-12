#pragma once

#include <ATen/ATen.h>
#include <oneDNN/Runtime.h>
#include <oneDNN/Utils.h>
#include <runtime/Utils.h>
#include <tensor/Context.h>
#include <utils/LRUCache.h>

#include <oneapi/dnnl/dnnl.hpp>

using namespace xpu::dpcpp;

namespace xpu {
namespace oneDNN {
namespace {

static inline memory::dims deconv_dst_size(
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

static inline memory::format_tag deconv_dat_fomt(int64_t ndim) {
  return (ndim == 4)
      ? memory::format_tag::nchw
      : ((ndim == 5) ? memory::format_tag::ncdhw : memory::format_tag::undef);
}

static inline memory::format_tag deconv_wgh_fmt(
    int64_t ndim,
    bool grouped = false) {
  return (ndim == 4)
      ? (grouped ? memory::format_tag::giohw : memory::format_tag::iohw)
      : ((ndim == 5) ? (grouped ? memory::format_tag::giodhw
                                : memory::format_tag::iodhw)
                     : memory::format_tag::undef);
}

static inline memory::dims deconv_compatible_dilation(IntArrayRef& dilation) {
  memory::dims ret = dilation.vec();
  for (auto it = ret.begin(); it != ret.end(); it++) {
    *it -= 1;
  }
  return ret;
}

static inline memory::dims deconv_compatible_wgh_dims(
    int64_t ndim,
    int64_t groups,
    int64_t oc,
    int64_t ic,
    IntArrayRef wgh_tz) {
  if (ndim == 4) {
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

static Tensor deconvolution(
    const Tensor& src_r,
    const Tensor& wgh_r,
    const Tensor& bia,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dst_padding,
    IntArrayRef dilation,
    int64_t groups) {
  auto src = src_r.contiguous();
  auto wgh = wgh_r.contiguous();
  bool with_bias = bia.defined();
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = deconv_compatible_dilation(dilation);

  auto ndim = src.ndimension();
  auto dst_tz = deconv_dst_size(
      src.sizes(), wgh.sizes(), padding, stride, dilation, dst_padding, groups);
  auto ic = src.size(1);
  auto oc = dst_tz[1];
  memory::dims src_tz = src.sizes().vec();
  memory::dims wgh_tz =
      deconv_compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());
  memory::dims bia_tz = {oc};
  auto dst = at::empty(dst_tz, src.options());

  memory::data_type src_dt = get_onednn_dtype(src);
  memory::data_type wgh_dt = get_onednn_dtype(wgh);
  memory::data_type dst_dt = get_onednn_dtype(dst);
  memory::data_type bia_dt =
      with_bias ? get_onednn_dtype(bia) : memory::data_type::undef;

  memory::format_tag src_fmt = deconv_dat_fomt(ndim);
  memory::format_tag wgh_fmt = deconv_wgh_fmt(ndim, groups != 1);
  memory::format_tag dst_fmt = src_fmt;
  memory::format_tag bia_fmt = memory::format_tag::x;

  auto src_md = memory::desc(src_tz, src_dt, src_fmt);
  auto wgh_md = memory::desc(wgh_tz, wgh_dt, wgh_fmt);
  auto dst_md = memory::desc(dst_tz, dst_dt, dst_fmt);
  auto bia_md =
      with_bias ? memory::desc(bia_tz, bia_dt, bia_fmt) : memory::desc();

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key_pd;
  create_key(
      key_pd,
      src_md,
      wgh_md,
      with_bias,
      dst_dt,
      _stride,
      _dilation,
      _padding,
      _padding);
#endif

  auto deconv_fwd_desc = with_bias ? deconvolution_forward::desc(
                                         prop_kind::forward,
                                         algorithm::deconvolution_direct,
                                         src_md,
                                         wgh_md,
                                         bia_md,
                                         dst_md,
                                         _stride,
                                         _dilation,
                                         _padding,
                                         _padding)
                                   : deconvolution_forward::desc(
                                         prop_kind::forward,
                                         algorithm::deconvolution_direct,
                                         src_md,
                                         wgh_md,
                                         bia_md,
                                         dst_md,
                                         _stride,
                                         _dilation,
                                         _padding,
                                         _padding);

  auto deconv_fwd_pd =
      deconvolution_forward::primitive_desc(deconv_fwd_desc, engine);

  auto src_m = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
  auto wgh_m = dpcpp_onednn_memory(wgh_md, engine, wgh.data_ptr());
  auto dst_m = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());

  std::unordered_map<int, memory> args = {
      {DNNL_ARG_SRC, src_m}, {DNNL_ARG_WEIGHTS, wgh_m}, {DNNL_ARG_DST, dst_m}};

  if (with_bias) {
    auto bia_m = dpcpp_onednn_memory(bia_md, engine, bia.data_ptr());
    args.insert({DNNL_ARG_BIAS, bia_m});
  }

#ifdef USE_PRIMITIVE_CACHE
  auto deconv_fwd =
      fetch_or_create_m<deconvolution_forward>(key_pd, deconv_fwd_pd);
#else
  auto deconv_fwd = deconvolution_forward(deconv_fwd_pd);
#endif

  DPCPP_ONEDNN_EXEC(deconv_fwd, strm, args);

  return dst;
}

static Tensor deconvolution_backward_data(
    const Tensor& src_r,
    const Tensor& wgh_r,
    const Tensor& diff_dst_r,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool with_bias) {
  auto src = src_r.contiguous();
  auto wgh = wgh_r.contiguous();
  auto diff_dst = diff_dst_r.contiguous();
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = deconv_compatible_dilation(dilation);

  auto ndim = src.ndimension();
  auto diff_src = at::empty(src.sizes(), diff_dst.options());

  auto ic = src.size(1);
  auto oc = diff_dst.size(1);
  memory::dims src_tz = src.sizes().vec();
  memory::dims wgh_tz =
      deconv_compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());
  memory::dims dst_tz = diff_dst.sizes().vec();
  memory::dims bia_tz = {oc};

  memory::data_type src_dt = get_onednn_dtype(src);
  memory::data_type wgh_dt = get_onednn_dtype(wgh);
  memory::data_type dst_dt = src_dt;
  memory::data_type bia_dt =
      with_bias ? get_onednn_dtype(wgh) : memory::data_type::undef;

  memory::format_tag src_fmt = deconv_dat_fomt(ndim);
  memory::format_tag wgh_fmt = deconv_wgh_fmt(ndim, groups != 1);
  memory::format_tag dst_fmt = src_fmt;
  memory::format_tag bia_fmt = memory::format_tag::x;

  auto src_md = memory::desc(src_tz, src_dt, src_fmt);
  auto wgh_md = memory::desc(wgh_tz, wgh_dt, wgh_fmt);
  auto dst_md = memory::desc(dst_tz, dst_dt, dst_fmt);
  auto bia_md =
      with_bias ? memory::desc(bia_tz, bia_dt, bia_fmt) : memory::desc();
  auto diff_dst_md = dst_md;
  auto diff_src_md = src_md;

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key_pd;
  create_key(
      key_pd,
      src_md,
      wgh_md,
      with_bias,
      dst_dt,
      _stride,
      _dilation,
      _padding,
      _padding);
#endif

  auto deconv_fwd_desc = with_bias ? deconvolution_forward::desc(
                                         prop_kind::forward,
                                         algorithm::deconvolution_direct,
                                         src_md,
                                         wgh_md,
                                         bia_md,
                                         dst_md,
                                         _stride,
                                         _dilation,
                                         _padding,
                                         _padding)
                                   : deconvolution_forward::desc(
                                         prop_kind::forward,
                                         algorithm::deconvolution_direct,
                                         src_md,
                                         wgh_md,
                                         dst_md,
                                         _stride,
                                         _dilation,
                                         _padding,
                                         _padding);

  auto deconv_fwd_pd =
      deconvolution_forward::primitive_desc(deconv_fwd_desc, engine);

  auto deconv_bwd_d_desc = deconvolution_backward_data::desc(
      algorithm::deconvolution_direct,
      src_md,
      wgh_md,
      dst_md,
      _stride,
      _dilation,
      _padding,
      _padding);

  auto deconv_bwd_d_pd = deconvolution_backward_data::primitive_desc(
      deconv_bwd_d_desc, engine, deconv_fwd_pd);

  auto diff_dst_m =
      dpcpp_onednn_memory(diff_dst_md, engine, diff_dst.data_ptr());
  auto wgh_m = dpcpp_onednn_memory(wgh_md, engine, wgh.data_ptr());
  auto diff_src_m =
      dpcpp_onednn_memory(diff_src_md, engine, diff_src.data_ptr());

#ifdef USE_PRIMITIVE_CACHE
  auto deconv_bwd_d = fetch_or_create_m<dnnl::deconvolution_backward_data>(
      key_pd, deconv_bwd_d_pd);
#else
  auto deconv_bwd_d = dnnl::deconvolution_backward_data(deconv_bwd_d_pd);
#endif
  DPCPP_ONEDNN_EXEC(
      deconv_bwd_d,
      strm,
      {{DNNL_ARG_DIFF_DST, diff_dst_m},
       {DNNL_ARG_WEIGHTS, wgh_m},
       {DNNL_ARG_DIFF_SRC, diff_src_m}});

  return diff_src;
}

static std::tuple<at::Tensor, at::Tensor> deconvolution_backward_weights(
    const Tensor& src_r,
    const Tensor& wgh_r,
    const Tensor& diff_dst_r,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool with_bias) {
  auto src = src_r.contiguous();
  auto wgh = wgh_r.contiguous();
  auto diff_dst = diff_dst_r.contiguous();
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  memory::dims _stride = stride.vec();
  memory::dims _padding = padding.vec();
  memory::dims _dilation = deconv_compatible_dilation(dilation);

  auto ndim = src.ndimension();
  auto diff_wgh = at::empty(wgh.sizes(), diff_dst.options());
  auto diff_bia =
      with_bias ? at::empty({diff_dst.size(1)}, diff_dst.options()) : Tensor();

  auto ic = src.size(1);
  auto oc = diff_dst.size(1);
  memory::dims src_tz = src.sizes().vec();
  memory::dims wgh_tz =
      deconv_compatible_wgh_dims(ndim, groups, oc, ic, wgh.sizes());
  memory::dims dst_tz = diff_dst.sizes().vec();
  memory::dims bia_tz = {oc};

  memory::data_type src_dt = get_onednn_dtype(src);
  memory::data_type wgh_dt = get_onednn_dtype(wgh);
  memory::data_type dst_dt = src_dt;
  memory::data_type bia_dt =
      with_bias ? get_onednn_dtype(wgh) : memory::data_type::f32;

  memory::format_tag src_fmt = deconv_dat_fomt(ndim);
  memory::format_tag wgh_fmt = deconv_wgh_fmt(ndim, groups != 1);
  memory::format_tag dst_fmt = src_fmt;
  memory::format_tag bia_fmt = memory::format_tag::x;

  auto src_md = memory::desc(src_tz, src_dt, src_fmt);
  auto wgh_md = memory::desc(wgh_tz, wgh_dt, wgh_fmt);
  auto dst_md = memory::desc(dst_tz, dst_dt, dst_fmt);
  auto bia_md =
      with_bias ? memory::desc(bia_tz, bia_dt, bia_fmt) : memory::desc();
  auto diff_dst_md = dst_md;
  auto diff_wgh_md = wgh_md;
  auto diff_bia_md = bia_md;

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key_pd;
  create_key(
      key_pd,
      src_md,
      wgh_md,
      with_bias,
      dst_dt,
      _stride,
      _dilation,
      _padding,
      _dilation);
#endif

  auto deconv_fwd_desc = with_bias ? deconvolution_forward::desc(
                                         prop_kind::forward,
                                         algorithm::deconvolution_direct,
                                         src_md,
                                         wgh_md,
                                         bia_md,
                                         dst_md,
                                         _stride,
                                         _dilation,
                                         _padding,
                                         _padding)
                                   : deconvolution_forward::desc(
                                         prop_kind::forward,
                                         algorithm::deconvolution_direct,
                                         src_md,
                                         wgh_md,
                                         dst_md,
                                         _stride,
                                         _dilation,
                                         _padding,
                                         _padding);

  auto deconv_fwd_pd =
      deconvolution_forward::primitive_desc(deconv_fwd_desc, engine);

  auto deconv_bwd_w_desc = with_bias ? deconvolution_backward_weights::desc(
                                           algorithm::deconvolution_direct,
                                           src_md,
                                           wgh_md,
                                           bia_md,
                                           dst_md,
                                           _stride,
                                           _dilation,
                                           _padding,
                                           _padding)
                                     : deconvolution_backward_weights::desc(
                                           algorithm::deconvolution_direct,
                                           src_md,
                                           wgh_md,
                                           dst_md,
                                           _stride,
                                           _dilation,
                                           _padding,
                                           _padding);

  auto deconv_bwd_w_pd = deconvolution_backward_weights::primitive_desc(
      deconv_bwd_w_desc, engine, deconv_fwd_pd);

  auto src_m = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
  auto diff_dst_m =
      dpcpp_onednn_memory(diff_dst_md, engine, diff_dst.data_ptr());
  auto diff_wgh_m =
      dpcpp_onednn_memory(diff_wgh_md, engine, diff_wgh.data_ptr());

#ifdef USE_PRIMITIVE_CACHE
  auto deconv_bwd_w = fetch_or_create_m<dnnl::deconvolution_backward_weights>(
      key_pd, deconv_bwd_w_pd);
#else
  auto deconv_bwd_w = dnnl::deconvolution_backward_weights(deconv_bwd_w_pd);
#endif

  std::unordered_map<int, memory> args = {
      {DNNL_ARG_DIFF_DST, diff_dst_m},
      {DNNL_ARG_SRC, src_m},
      {DNNL_ARG_DIFF_WEIGHTS, diff_wgh_m}};

  if (with_bias) {
    auto diff_bia_m =
        dpcpp_onednn_memory(diff_bia_md, engine, diff_bia.data_ptr());
    args.insert({DNNL_ARG_DIFF_BIAS, diff_bia_m});
  }

  DPCPP_ONEDNN_EXEC(deconv_bwd_w, strm, args);

  return std::tuple<at::Tensor, at::Tensor>{diff_wgh, diff_bia};
}
} // namespace oneDNN
} // namespace xpu
