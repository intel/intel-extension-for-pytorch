#pragma once

#include <ATen/ATen.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>
#include <tensor/Context.h>
#include "Utils.h"
#include "Reorder.h"

#include <oneapi/dnnl/dnnl.hpp>

#ifdef USE_PRIMITIVE_CACHE
#include <oneDNN/LRUCache.h>
#endif


using namespace dnnl;
using namespace at::AtenIpexTypeXPU;

namespace xpu {
namespace oneDNN {

static std::tuple<Tensor, Tensor, Tensor> layer_norm(
    const Tensor& src,
    const Tensor& wgh,
    const Tensor& bia,
    double epsilon) {
  auto engine =
      GpuEngineManager::Instance().get_engine(Device(kXPU, current_device()));
  auto strm = GpuStreamManager::Instance().get_stream();

  // FP16 Data Type only support forward_inference
  bool training = src.scalar_type() == ScalarType::Half ? false : true;
  auto prop = training ?
              prop_kind::forward_training :
              prop_kind::forward_inference;
  normalization_flags flags = normalization_flags::use_scale_shift;
  bool useScaleShift = (bool)(flags & normalization_flags::use_scale_shift);

  int32_t n, ic, ih;
  memory::dims tz, st, stats_tz;
  memory::format_tag stats_fmt;
  if (src.ndimension() == 3) {
    n = src.size(0);
    ic = src.size(1);
    ih = src.size(2);
    tz = {n, ic, ih};
    st = {src.stride(0), src.stride(1), src.stride(2)};
    stats_tz = {n, ic};
    stats_fmt = memory::format_tag::ab;
  } else {
    ic = src.size(0);
    ih = src.size(1);
    tz = {ic, ih};
    st = {src.stride(0), src.stride(1)};
    stats_tz = {ic};
    stats_fmt = memory::format_tag::a;
  }

  memory::data_type dt = get_onednn_dtype(src);
  memory::data_type stats_dt = memory::data_type::f32;

  auto src_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src);
  auto md = src_ctx.is_plain() ? memory::desc({tz}, dt, {st}) : src_ctx.meta();
  auto stats_md = memory::desc(stats_tz, stats_dt, stats_fmt);
  auto dst = src_ctx.is_plain() ?
             at::empty_like(src, src.options()) :
             empty_opaque_tensor(md, src.options(), c10::nullopt);

  auto src_m = dpcpp_onednn_memory(md, engine, src.data_ptr());
  auto dst_m = dpcpp_onednn_memory(md, engine, dst.data_ptr());

  auto ln_fwd_desc = training ?
      layer_normalization_forward::desc(prop, md, stats_md, epsilon, flags) :
      layer_normalization_forward::desc(prop, md, epsilon, flags);
  auto ln_fwd_pd =
      layer_normalization_forward::primitive_desc(ln_fwd_desc, engine);

  std::unordered_map<int, memory> args = {
      {DNNL_ARG_SRC, src_m},
      {DNNL_ARG_DST, dst_m},
  };

  Tensor mean, rstd;
  auto stats_exp_md = ln_fwd_pd.mean_desc();
  if (training) {
    auto stats_usr_md = memory::desc(stats_tz, stats_dt, stats_fmt);
    if (!src_ctx.is_plain() && stats_exp_md != stats_usr_md) {
      mean = empty_opaque_tensor(stats_exp_md, wgh.options(), c10::nullopt);
      rstd = empty_opaque_tensor(stats_exp_md, wgh.options(), c10::nullopt);
    } else {
      mean = at::empty(stats_tz, wgh.options());
      rstd = at::empty(stats_tz, wgh.options());
    }

    auto mean_memory = dpcpp_onednn_memory(stats_exp_md, engine, mean.data_ptr());
    auto var_memory = dpcpp_onednn_memory(stats_exp_md, engine, rstd.data_ptr());

    args.insert({DNNL_ARG_MEAN, mean_memory});
    args.insert({DNNL_ARG_VARIANCE, var_memory});
  }

  // LayerNorm weight/bias only support FP32 data type in oneDNN
  Tensor wgh_ = at::empty_like(wgh, at::kFloat);
  Tensor bia_ = at::empty_like(bia, at::kFloat);
  if (wgh.scalar_type() == ScalarType::Half) {
    dpcppMemoryCopyType(wgh_.data_ptr<float>(), wgh.data_ptr<at::Half>(), ih);
    dpcppMemoryCopyType(bia_.data_ptr<float>(), bia.data_ptr<at::Half>(), ih);
  } else if (wgh.scalar_type() == ScalarType::BFloat16) {
    dpcppMemoryCopyType(wgh_.data_ptr<float>(), wgh.data_ptr<at::BFloat16>(), ih);
    dpcppMemoryCopyType(bia_.data_ptr<float>(), bia.data_ptr<at::BFloat16>(), ih);
  } else {
    wgh_ = wgh;
    bia_ = bia;
  }

  if (useScaleShift) {
    auto wgh_bia = at::empty(2 * ih, wgh_.options());
    dpcppMemcpyAsync(
        wgh_bia.data_ptr(),
        wgh_.data_ptr(),
        ih * wgh_.itemsize(),
        DeviceToDevice);
    dpcppMemcpyAsync(
        static_cast<uint8_t*>(wgh_bia.data_ptr()) + ih * wgh_.itemsize(),
        bia_.data_ptr(),
        ih * bia_.itemsize(),
        DeviceToDevice);

    auto wgh_bia_m = dpcpp_onednn_memory(
        ln_fwd_pd.weights_desc(), engine, wgh_bia.data_ptr());
    args.insert({DNNL_ARG_SCALE_SHIFT, wgh_bia_m});
  }

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  if (training)
    create_key(key, md, stats_exp_md, epsilon, flags);
  else
    create_key(key, md, epsilon, flags);
  auto ln_fwd = fetch_or_create_m<layer_normalization_forward>(key, ln_fwd_pd);
#else
  auto ln_fwd = layer_normalization_forward(ln_fwd_pd);
#endif

  DPCPP_ONEDNN_EXEC(ln_fwd, strm, args);
  return std::make_tuple(dst, mean, rstd);
}

static std::tuple<Tensor, Tensor, Tensor> layer_norm_backward(
    const Tensor& diff_dst,
    const Tensor& src,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& wgh,
    double epsilon) {
  auto engine =
      GpuEngineManager::Instance().get_engine(Device(kXPU, current_device()));
  auto strm = GpuStreamManager::Instance().get_stream();

  auto flags = normalization_flags::use_scale_shift;
  bool useScaleShift = (bool)(flags & normalization_flags::use_scale_shift);

  int32_t n, ic, ih;
  memory::dims src_tz, src_st, diff_dst_tz, diff_dst_st, stats_tz;
  memory::format_tag stats_fmt;
  if (src.ndimension() == 3) {
    n = src.size(0);
    ic = src.size(1);
    ih = src.size(2);
    src_tz = {n, ic, ih};
    src_st = {src.stride(0), src.stride(1), src.stride(2)};
    diff_dst_tz = {n, ic, ih};
    diff_dst_st = {diff_dst.stride(0), diff_dst.stride(1), diff_dst.stride(2)};
    stats_tz = {n, ic};
    stats_fmt = memory::format_tag::ab;
  } else {
    ic = src.size(0);
    ih = src.size(1);
    src_tz = {ic, ih};
    src_st = {src.stride(0), src.stride(1)};
    diff_dst_tz = {ic, ih};
    diff_dst_st = {diff_dst.stride(0), diff_dst.stride(1)};
    stats_tz = {ic};
    stats_fmt = memory::format_tag::a;
  }

  memory::data_type dt = get_onednn_dtype(src);
  memory::data_type stats_dt = memory::data_type::f32;

  auto src_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src);
  auto diff_dst_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(diff_dst);
  auto mean_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(mean);
  auto rstd_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(rstd);

  auto src_md = src_ctx.is_plain() ?
      memory::desc({src_tz}, dt, {src_st}) : src_ctx.meta();
  auto diff_dst_md = diff_dst_ctx.is_plain() ?
      memory::desc({diff_dst_tz}, dt, {diff_dst_st}) : diff_dst_ctx.meta();
  auto exp_md = !src_ctx.is_plain() ? src_md : diff_dst_md;
  auto mean_md = mean_ctx.is_plain() ?
      memory::desc({stats_tz}, stats_dt, stats_fmt) : mean_ctx.meta();
  auto rstd_md = rstd_ctx.is_plain() ?
      memory::desc({stats_tz}, stats_dt, stats_fmt) : rstd_ctx.meta();

  auto ln_fwd_desc = layer_normalization_forward::desc(
      dnnl::prop_kind::forward_training, exp_md, mean_md, epsilon, flags);
  auto ln_fwd_pd = layer_normalization_forward::primitive_desc(
      ln_fwd_desc, engine);

  auto ln_bwd_desc = layer_normalization_backward::desc(
      dnnl::prop_kind::backward, exp_md, exp_md, mean_md, epsilon, flags);
  auto ln_bwd_pd = layer_normalization_backward::primitive_desc(
      ln_bwd_desc, engine, ln_fwd_pd);

  Tensor src_;
  memory src_m = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
  if (src_md != exp_md) {
    src_ = empty_opaque_tensor(exp_md, src.options(), c10::nullopt);
    xpu::oneDNN::reorder(src, src_);
    src_m = dpcpp_onednn_memory(exp_md, engine, src_.data_ptr());
  }

  Tensor diff_dst_;
  memory diff_dst_m = dpcpp_onednn_memory(diff_dst_md, engine, diff_dst.data_ptr());
  if (diff_dst_md != exp_md) {
    diff_dst_ = empty_opaque_tensor(exp_md, diff_dst.options(), c10::nullopt);
    xpu::oneDNN::reorder(diff_dst, diff_dst_);
    diff_dst_m = dpcpp_onednn_memory(exp_md, engine, diff_dst_.data_ptr());
  }

  Tensor diff_src = !src_ctx.is_plain() || !diff_dst_ctx.is_plain() ?
      empty_opaque_tensor(exp_md, src.options(), c10::nullopt) :
      at::empty_like(src);
  memory diff_src_m = dpcpp_onednn_memory(exp_md, engine, diff_src.data_ptr());

  auto stats_exp_md = ln_bwd_pd.mean_desc();
  Tensor mean_;
  memory mean_m = dpcpp_onednn_memory(mean_md, engine, mean.data_ptr());
  if (mean_md != stats_exp_md) {
    mean_ = empty_opaque_tensor(stats_exp_md, mean.options(), c10::nullopt);
    xpu::oneDNN::reorder(mean, mean_);
    mean_m = dpcpp_onednn_memory(stats_exp_md, engine, mean_.data_ptr());
  }

  Tensor rstd_;
  memory rstd_m = dpcpp_onednn_memory(rstd_md, engine, rstd.data_ptr());
  if (rstd_md != stats_exp_md) {
    rstd_ = empty_opaque_tensor(stats_exp_md, rstd.options(), c10::nullopt);
    xpu::oneDNN::reorder(rstd, rstd_);
    rstd_m = dpcpp_onednn_memory(stats_exp_md, engine, rstd_.data_ptr());
  }

  std::unordered_map<int, memory> args = {
      {DNNL_ARG_SRC,        src_m       },
      {DNNL_ARG_DIFF_DST,   diff_dst_m  },
      {DNNL_ARG_MEAN,       mean_m      },
      {DNNL_ARG_VARIANCE,   rstd_m      },
      {DNNL_ARG_DIFF_SRC,   diff_src_m  },
  };
  Tensor diff_wgh_bia;
  if (useScaleShift) {
    auto wgh_bia = at::empty(2 * ih, wgh.options());
    dpcppMemcpyAsync(
        wgh_bia.data_ptr(),
        wgh.data_ptr(),
        ih * wgh.itemsize(),
        DeviceToDevice);
    dpcppMemsetAsync(
        static_cast<uint8_t*>(wgh_bia.data_ptr()) + ih * sizeof(float),
        0,
        ih * wgh.itemsize());
    auto wgh_bia_m = dpcpp_onednn_memory(
        ln_bwd_pd.weights_desc(), engine, wgh_bia.data_ptr());

    diff_wgh_bia = at::empty(2 * ih, wgh.options());
    auto diff_wgh_bia_m = dpcpp_onednn_memory(
        ln_bwd_pd.diff_weights_desc(), engine, diff_wgh_bia.data_ptr());

    args.insert({DNNL_ARG_SCALE_SHIFT,      wgh_bia_m       });
    args.insert({DNNL_ARG_DIFF_SCALE_SHIFT, diff_wgh_bia_m  });
  }

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, src_md, diff_dst_md, epsilon, flags);
  auto ln_backward = fetch_or_create_m<layer_normalization_backward>(key, ln_bwd_pd);
#else
  auto ln_backward = layer_normalization_backward(ln_bwd_pd);
#endif
  DPCPP_ONEDNN_EXEC(ln_backward, strm, args);

  Tensor diff_wgh = at::empty_like(wgh);
  Tensor diff_bia = at::empty_like(wgh);
  if (useScaleShift) {
    dpcppMemcpyAsync(
        diff_wgh.data_ptr(),
        diff_wgh_bia.data_ptr(),
        ih * wgh.itemsize(),
        DeviceToDevice);
    dpcppMemcpyAsync(
        diff_bia.data_ptr(),
        static_cast<uint8_t*>(diff_wgh_bia.data_ptr()) + ih * wgh.itemsize(),
        ih * wgh.itemsize(),
        DeviceToDevice);
  }

  return std::make_tuple(diff_src, diff_wgh, diff_bia);
}

}}
