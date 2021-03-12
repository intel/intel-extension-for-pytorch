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
namespace dpcpp {
namespace oneDNN {

static inline void reorder(const Tensor& src, Tensor& dst,
                           const primitive_attr& pattr = primitive_attr()) {
  TORCH_CHECK(dst.data_ptr() != src.data_ptr(),
             "oneDNN reorder supports out-place implementation only ...");

  auto engine = GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  memory::dims src_tz = src.sizes().vec();
  memory::dims dst_tz = dst.sizes().vec();
  auto src_dt = dt_to_dnnl(src.scalar_type());
  auto dst_dt = dt_to_dnnl(dst.scalar_type());
  auto src_fmt = get_dnnl_default_format(src.ndimension());
  auto dst_fmt = get_dnnl_default_format(dst.ndimension());

  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  memory::desc src_desc = src_ctx.is_plain() ?
                          memory::desc(src_tz, src_dt, src_fmt) :
                          src_ctx.meta();
  auto src_mem = dpcpp_onednn_memory(src_desc, engine, src.data_ptr());

  auto dst_ctx = DPCPPTensorContext::get_tensor_ctx(dst);
  memory::desc dst_desc = dst_ctx.is_plain() ?
                          memory::desc(dst_tz, dst_dt, dst_fmt) :
                          dst_ctx.meta();
  auto dst_mem = dpcpp_onednn_memory(dst_desc, engine, dst.data_ptr());

  std::vector<float> oscale;
  int mask = src_ctx.scales().size() > 1 ? ONEDNN_SCALES_MASK_BY_CHANNEL(0) : 0;
  pattr.get_output_scales(mask, oscale);

  primitive prim;
  if (!oscale.empty()) {
#ifdef USE_PRIMITIVE_CACHE
    lru_key_t key;
    create_key(key, src_desc, dst_desc, oscale);
    prim = fetch_or_create_m<dnnl::reorder>(key, src_mem, dst_mem, pattr);
#else
    prim = dnnl::reorder(src_mem, dst_mem, pattr);
#endif
  } else {
#ifdef USE_PRIMITIVE_CACHE
    lru_key_t key;
    create_key(key, src_desc, dst_desc);
    prim = fetch_or_create_m<dnnl::reorder>(key, src_mem, dst_mem);
#else
    prim = dnnl::reorder(src_mem, dst_mem);
#endif
  }

  DPCPP_ONEDNN_EXEC(prim, strm,
      {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
}

static inline Tensor reorder_copy(Tensor& dst, const Tensor& src) {
  auto engine = GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  // align to dst
  auto dst_ctx = DPCPPTensorContext::get_tensor_ctx(dst);
  memory::desc dst_desc = dst_ctx.is_plain() ?
                             memory::desc(get_onednn_dims(dst),
                                          get_onednn_dtype(dst),
                                          get_onednn_strides(dst)) :
                             dst_ctx.meta();
  memory dst_mem = dpcpp_onednn_memory(dst_desc, engine, dst.data_ptr());

  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  memory::desc src_desc = src_ctx.is_plain() ?
                          memory::desc(get_onednn_dims(src),
                                       get_onednn_dtype(src),
                                       get_onednn_strides(src)) :
                          src_ctx.meta();
  memory src_mem = dpcpp_onednn_memory(src_desc, engine, src.data_ptr());
  // simple copy checking address only
  if (dst.data_ptr() != src.data_ptr()) {
#ifdef USE_PRIMITIVE_CACHE
    lru_key_t key;
    create_key(key, src_desc, dst_desc);
    auto prim = fetch_or_create_m<dnnl::reorder>(key, src_mem, dst_mem);
#else
    auto prim = dnnl::reorder(src_mem, dst_mem);
#endif
    DPCPP_ONEDNN_EXEC(prim, strm, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
  }

  return dst;
}

}}}
