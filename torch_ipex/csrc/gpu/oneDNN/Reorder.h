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
using namespace at::dpcpp::oneDNN; // workaround

namespace at {
namespace xpu {
namespace oneDNN {

struct ReorderAttr {
public:
  ReorderAttr(bool is_group = false)
      : pattr_(primitive_attr()),
        sc_(std::vector<float>()) {}

public:
  void set_src_sc_and_zp(int scmask,
                         std::vector<float> sc,
                         int zpmask,
                         std::vector<int> zp) {
    pattr_.set_output_scales(scmask, sc);
    pattr_.set_zero_points(DNNL_ARG_SRC, zpmask, zp);
    sc_ = sc;
  }

  void set_dst_sc_and_zp(int scmask,
                         std::vector<float> sc,
                         int zpmask,
                         std::vector<int> zp) {
    pattr_.set_output_scales(scmask, sc);
    pattr_.set_zero_points(DNNL_ARG_DST, zpmask, zp);
    sc_ = sc;
  }

  bool is_quant() const { return !sc_.empty(); }

  std::vector<float> sc() const { return sc_; }

  primitive_attr pattr() const { return pattr_; }

private:
  primitive_attr pattr_;
  std::vector<float> sc_;
};

static inline void reorder(const Tensor& src, Tensor& dst,
                           const ReorderAttr& rattr = ReorderAttr()) {
  TORCH_CHECK(dst.data_ptr() != src.data_ptr(),
             "oneDNN reorder supports out-place implementation only ...");

  auto engine = GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto check_group_and_create_plain_md =
      [](const Tensor& src, const Tensor& dst) -> memory::desc {
        if (src.ndimension() == dst.ndimension()) {
          return memory::desc(get_onednn_dims(src),
                              get_onednn_dtype(src),
                              get_onednn_strides(src));
        } else if ((src.ndimension() == dst.ndimension() - 1) &&
                   (src.size(0) == dst.size(0) * dst.size(1))) {
          // group tensor
          return memory::desc(get_onednn_dims(dst),
                              get_onednn_dtype(src),
                              get_onednn_strides(dst.contiguous()));
        } else {
          TORCH_CHECK(0, "invalid src/dst dimension in oneDNN reorder ...");
        }
      };

  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  memory::desc src_md = src_ctx.is_plain() ?
      check_group_and_create_plain_md(src, dst) :
      src_ctx.meta();
  auto src_mem = dpcpp_onednn_memory(src_md, engine, src.data_ptr());

  auto dst_ctx = DPCPPTensorContext::get_tensor_ctx(dst);
  memory::desc dst_md = dst_ctx.is_plain() ?
      memory::desc(get_onednn_dims(dst),
                   get_onednn_dtype(dst),
                   get_onednn_strides(dst)) :
      dst_ctx.meta();
  auto dst_mem = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());

  primitive prim;
  if (rattr.is_quant()) {
  auto pattr = rattr.pattr();
#ifdef USE_PRIMITIVE_CACHE
    lru_key_t key;
    auto oscale = rattr.sc();
    create_key(key, src_md, dst_md, oscale);
    prim = fetch_or_create_m<dnnl::reorder>(key, src_mem, dst_mem, pattr);
#else
    prim = dnnl::reorder(src_mem, dst_mem, pattr);
#endif
  } else {
#ifdef USE_PRIMITIVE_CACHE
    lru_key_t key;
    create_key(key, src_md, dst_md);
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
  memory::desc dst_md = dst_ctx.is_plain() ?
                        memory::desc(get_onednn_dims(dst),
                                     get_onednn_dtype(dst),
                                     get_onednn_strides(dst)) :
                        dst_ctx.meta();
  memory dst_mem = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());

  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  memory::desc src_md = src_ctx.is_plain() ?
                        memory::desc(get_onednn_dims(src),
                                     get_onednn_dtype(src),
                                     get_onednn_strides(src)) :
                        src_ctx.meta();
  memory src_mem = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
  // simple copy checking address only
  if (dst.data_ptr() != src.data_ptr()) {
#ifdef USE_PRIMITIVE_CACHE
    lru_key_t key;
    create_key(key, src_md, dst_md);
    auto prim = fetch_or_create_m<dnnl::reorder>(key, src_mem, dst_mem);
#else
    auto prim = dnnl::reorder(src_mem, dst_mem);
#endif
    DPCPP_ONEDNN_EXEC(prim, strm, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
  }

  return dst;
}

}}}
