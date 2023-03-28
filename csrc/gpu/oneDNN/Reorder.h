#pragma once

#include <ATen/ATen.h>

#include <ATen/record_function.h>
#include <oneDNN/Runtime.h>
#include <runtime/Utils.h>
#include <tensor/Context.h>
#include <utils/LRUCache.h>
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace at::AtenIpexTypeXPU;

namespace xpu {
namespace oneDNN {

struct ReorderAttr {
 public:
  ReorderAttr(bool is_group = false) : pattr_(primitive_attr()) {}

 public:
  // [Note: Scale setting for reorder]
  // For no post op on reorder, dst = src_scale * src / dst_scale;
  // dst_scale should be set carefully.
  void set_src_sc_and_zp_mask(int mask) {
    pattr_.set_scales_mask(DNNL_ARG_SRC, mask);
    pattr_.set_zero_points_mask(DNNL_ARG_SRC, mask);
  }

  void set_dst_sc_and_zp_mask(int mask) {
    pattr_.set_scales_mask(DNNL_ARG_DST, mask);
    pattr_.set_zero_points_mask(DNNL_ARG_DST, mask);
  }

  primitive_attr pattr() const {
    return pattr_;
  }

 private:
  primitive_attr pattr_;
};

static inline memory::desc check_group_and_create_plain_md(
    const Tensor& src,
    const Tensor& dst) {
  if (src.ndimension() == dst.ndimension()) {
    return memory::desc(
        get_onednn_dims(src),
        get_onednn_dtype_include_double(src),
        get_onednn_strides(src));
  } else if (
      ((src.ndimension() == dst.ndimension() - 1) &&
       (src.size(0) == dst.size(0) * dst.size(1))) ||
      ((src.ndimension() == dst.ndimension() + 1) &&
       (dst.size(0) == src.size(0) * src.size(1)))) {
    // group tensor
    return memory::desc(
        get_onednn_dims(dst),
        get_onednn_dtype_include_double(src),
        get_onednn_strides(dst.contiguous()));
  } else {
    TORCH_CHECK(0, "invalid src/dst dimension in oneDNN reorder ...");
  }
}

static inline void reorder(
    const Tensor& src,
    Tensor& dst,
    const ReorderAttr& rattr = ReorderAttr()) {
  RECORD_FUNCTION("dnnl_reorder", std::vector<c10::IValue>({src}));

  if (dst.is_same(src))
    return;

  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  memory::desc src_md = src_ctx.is_plain()
      ? check_group_and_create_plain_md(src, dst)
      : src_ctx.meta();
  auto src_mem = dpcpp_onednn_memory(src_md, engine, src.data_ptr());

  auto dst_ctx = DPCPPTensorContext::get_tensor_ctx(dst);
  memory::desc dst_md = dst_ctx.is_plain()
      ? memory::desc(
            get_onednn_dims(dst),
            get_onednn_dtype_include_double(dst),
            get_onednn_strides(dst))
      : dst_ctx.meta();
  auto dst_mem = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());

  primitive prim;
  prim = dnnl::reorder(src_mem, dst_mem);

  DPCPP_ONEDNN_EXEC(
      prim, strm, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
}

static inline void quantized_reorder(
    const Tensor& src,
    Tensor& dst,
    const Tensor& scale,
    const Tensor& zero_point,
    const ReorderAttr& rattr = ReorderAttr()) {
  RECORD_FUNCTION("dnnl_qreorder_per_channel", std::vector<c10::IValue>({src}));
  if (dst.is_same(src))
    return;

  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  memory::desc src_md = src_ctx.is_plain()
      ? check_group_and_create_plain_md(src, dst)
      : src_ctx.meta();
  auto src_mem = dpcpp_onednn_memory(src_md, engine, src.data_ptr());

  auto dst_ctx = DPCPPTensorContext::get_tensor_ctx(dst);
  memory::desc dst_md = dst_ctx.is_plain()
      ? memory::desc(
            get_onednn_dims(dst),
            get_onednn_dtype_include_double(dst),
            get_onednn_strides(dst))
      : dst_ctx.meta();
  auto dst_mem = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());

  memory::desc scale_md = memory::desc(
      get_onednn_dims(scale),
      memory::data_type::f32,
      get_onednn_strides(scale));
  auto sc_mem = dpcpp_onednn_memory(scale_md, engine, scale.data_ptr());

  primitive prim;
  auto pattr = rattr.pattr();
#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  // Here change scale to scale_md
  create_key(key, src_md, dst_md, scale_md);
  prim = fetch_or_create_m<dnnl::reorder>(key, src_mem, dst_mem, pattr);
#else
  prim = dnnl::reorder(src_mem, dst_mem, pattr);
#endif

  std::unordered_map<int, memory> reorder_args;

  reorder_args.insert({DNNL_ARG_SRC, src_mem});
  reorder_args.insert({DNNL_ARG_DST, dst_mem});

  // Construct zp memory
  if (src.is_quantized()) {
    // Dequantize
    auto src_zp_md = memory::desc(
        get_onednn_dims(zero_point),
        memory::data_type::s32,
        memory::format_tag::x);
    auto src_zp_mem =
        dpcpp_onednn_memory(src_zp_md, engine, zero_point.data_ptr());
    reorder_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, sc_mem});
    reorder_args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zp_mem});
    DPCPP_ONEDNN_EXEC(prim, strm, reorder_args);
  } else {
    // Quantize
    auto dst_zp_md = memory::desc(
        get_onednn_dims(zero_point),
        memory::data_type::s32,
        get_onednn_strides(zero_point));
    auto dst_zp_mem =
        dpcpp_onednn_memory(dst_zp_md, engine, zero_point.data_ptr());
    reorder_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, sc_mem});
    reorder_args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zp_mem});
    DPCPP_ONEDNN_EXEC(prim, strm, reorder_args);
  }
}

static inline void reorder_copy(const Tensor& src, Tensor& dst) {
  RECORD_FUNCTION("reorder_copy", std::vector<c10::IValue>({src}));
  xpu::oneDNN::reorder(src, dst);
}

} // namespace oneDNN
} // namespace xpu
