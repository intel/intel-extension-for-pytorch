#pragma once

#include <ATen/ATen.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>
#include <tensor/Context.h>
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>

#ifdef USE_PRIMITIVE_CACHE
#include <oneDNN/LRUCache.h>
#endif

using namespace dnnl;
using namespace at::AtenIpexTypeXPU;

namespace at {
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
  auto prop = training ? prop_kind::forward_training
                              : prop_kind::forward_inference;
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

  memory::data_type dt = dt_to_dnnl(src.scalar_type());
  memory::data_type stats_dt = memory::data_type::f32;

  auto src_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src);
  auto md = src_ctx.is_plain() ?
            memory::desc({tz}, dt, {st}) : src_ctx.meta();
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
      mean = empty_opaque_tensor(stats_exp_md, src.options(), c10::nullopt);
      rstd = empty_opaque_tensor(stats_exp_md, src.options(), c10::nullopt);
    } else {
      mean = at::empty(stats_tz, src.options());
      rstd = at::empty(stats_tz, src.options());
    }

    auto mean_memory = dpcpp_onednn_memory(stats_exp_md, engine, mean.data_ptr());
    auto var_memory = dpcpp_onednn_memory(stats_exp_md, engine, rstd.data_ptr());

    args.insert({DNNL_ARG_MEAN, mean_memory});
    args.insert({DNNL_ARG_VARIANCE, var_memory});
  }

  // weight/bias need fp32 data type
  if (useScaleShift) {
    auto wgh_bia = at::empty(2 * ih, wgh.options());
    dpcppMemcpyAsync(
        wgh_bia.data_ptr(),
        wgh.data_ptr(),
        ih * wgh.itemsize(),
        DeviceToDevice);
    dpcppMemcpyAsync(
        static_cast<uint8_t*>(wgh_bia.data_ptr()) + ih * wgh.itemsize(),
        bia.data_ptr(),
        ih * bia.itemsize(),
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

}}}
