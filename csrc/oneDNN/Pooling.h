#pragma once

#include <ATen/ATen.h>

#include <runtime/Utils.h>
#include <oneDNN/Runtime.h>
#include <oneDNN/LRUCache.h>
#include <tensor/Context.h>
#include <operators/comm/Scalar.h>
#include "Utils.h"
#include "Reorder.h"

#include <oneapi/dnnl/dnnl.hpp>

using namespace at::AtenIpexTypeXPU;
using namespace at::AtenIpexTypeQuantizedXPU;

namespace xpu {
namespace oneDNN {

using alg = dnnl::algorithm;

template <alg alg_kind>
static at::Tensor pooling(
    at::Tensor& dst,
    const at::Tensor& src,
    int64_t nbatch,
    int64_t nInputPlane,
    int64_t srcDepth,
    int64_t srcHeight,
    int64_t srcWidth,
    int64_t dstDepth,
    int64_t dstHeight,
    int64_t dstWidth,
    int kD,
    int kH,
    int kW,
    int dD,
    int dH,
    int dW,
    int padD,
    int padH,
    int padW) {
  at::Device curDevice = at::Device(at::kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  prop_kind prop_kind = dnnl::prop_kind::forward_training;
  auto data_t = get_onednn_dtype(src);
  if (data_t == memory::data_type::f16 || data_t == memory::data_type::s8 ||
      data_t == memory::data_type::u8 || data_t == memory::data_type::s32) {
    prop_kind = dnnl::prop_kind::forward_inference;
  }

  memory::format_tag format;

  memory::dims src_tz;
  memory::dims dst_tz;
  memory::dims kernel;
  memory::dims stride;
  memory::dims padding;
  memory::format_tag format_any = memory::format_tag::any;

  if (srcDepth == 0) {
    format = src.is_contiguous(at::MemoryFormat::ChannelsLast) ?
             memory::format_tag::nhwc :
             memory::format_tag::nchw;
    src_tz = {nbatch, nInputPlane, srcHeight, srcWidth};
    dst_tz = {nbatch, nInputPlane, dstHeight, dstWidth};
    kernel = {kH, kW};
    stride = {dH, dW};
    padding = {padH, padW};
  } else {
    format = src.is_contiguous(at::MemoryFormat::ChannelsLast3d) ?
             memory::format_tag::ndhwc :
             memory::format_tag::ncdhw;
    src_tz = {nbatch, nInputPlane, srcDepth, srcHeight, srcWidth};
    dst_tz = {nbatch, nInputPlane, dstDepth, dstHeight, dstWidth};
    kernel = {kD, kH, kW};
    stride = {dD, dH, dW};
    padding = {padD, padH, padW};
  }

  auto src_md = memory::desc({src_tz}, data_t, format);
  auto dst_md = memory::desc({dst_tz}, data_t, format);
  auto dst_md_any = memory::desc({dst_tz}, data_t, format_any);

  if (Settings::I().is_onednn_layout_enabled()) {
    auto src_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src);
    src_md = src_ctx.is_plain() ?
             memory::desc({src_tz}, data_t, format) :
             src_ctx.meta();
  }

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, src_md, dst_md_any, stride, kernel, padding, padding, alg_kind);
#endif
  auto pooling_fwd_desc = pooling_forward::desc(
      prop_kind,
      alg_kind,
      src_md,
      dst_md_any,
      stride,
      kernel,
      padding,
      padding);

  auto pooling_fwd_pd = pooling_forward::primitive_desc(pooling_fwd_desc, engine);

  memory src_m, dst_m;
  if (!Settings::I().is_onednn_layout_enabled() ||
      src.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      src.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    src_m = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
    dst_m = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());
  } else {
    src_m = dpcpp_onednn_memory(src_md, engine, src.data_ptr());

    auto expected_dst_md = pooling_fwd_pd.dst_desc();
    if (expected_dst_md != dst_md) {
      // reallocate memory due to padding needed by oneDNN in some blk fmt
      if (src.is_quantized()) {
        auto quantizer = dpcpp_make_per_tensor_affine_quantizer(
            src.q_scale(),
            src.q_zero_point(),
            at::typeMetaToScalarType(src.options().dtype()));
        dst = empty_opaque_qtensor(expected_dst_md, c10::nullopt, quantizer);
      } else {
        dst = empty_opaque_tensor(expected_dst_md, src.options(), c10::nullopt);
      }
      dst_m = dpcpp_onednn_memory(expected_dst_md, engine, dst.data_ptr());
    } else {
      dst_m = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());
    }
  }

#ifdef USE_PRIMITIVE_CACHE
  auto pooling_fwd = fetch_or_create_m<pooling_forward>(key, pooling_fwd_pd);
#else
  auto pooling_fwd = pooling_forward(pooling_fwd_pd);
#endif

  DPCPP_ONEDNN_EXEC(
      pooling_fwd,
      strm,
      {{DNNL_ARG_SRC, src_m}, {DNNL_ARG_DST, dst_m}});

  return dst;
}

template <algorithm alg_kind>
static std::tuple<at::Tensor, at::Tensor> pooling(
    at::Tensor& dst,
    at::Tensor& idx,
    const at::Tensor& src,
    int64_t nbatch,
    int64_t nInputPlane,
    int64_t srcDepth,
    int64_t srcHeight,
    int64_t srcWidth,
    int64_t dstDepth,
    int64_t dstHeight,
    int64_t dstWidth,
    int kD,
    int kH,
    int kW,
    int dD,
    int dH,
    int dW,
    int padD,
    int padH,
    int padW) {
  at::Device curDevice = at::Device(at::kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  auto prop_kind = dnnl::prop_kind::forward_training;
  auto data_t = get_onednn_dtype(src);
  if (data_t == memory::data_type::f16 || data_t == memory::data_type::s8 ||
      data_t == memory::data_type::u8 || data_t == memory::data_type::s32) {
    prop_kind = dnnl::prop_kind::forward_inference;
  }

  memory::format_tag format;
  memory::dims src_tz;
  memory::dims dst_tz;
  memory::dims kernel;
  memory::dims stride;
  memory::dims padding;

  if (srcDepth == 0) {
    format = src.is_contiguous(at::MemoryFormat::ChannelsLast) ?
             memory::format_tag::nhwc :
             memory::format_tag::nchw;
    src_tz = {nbatch, nInputPlane, srcHeight, srcWidth};
    dst_tz = {nbatch, nInputPlane, dstHeight, dstWidth};
    kernel = {kH, kW};
    stride = {dH, dW};
    padding = {padH, padW};
  } else {
    format = src.is_contiguous(at::MemoryFormat::ChannelsLast3d) ?
             memory::format_tag::ndhwc :
             memory::format_tag::ncdhw;
    src_tz = {nbatch, nInputPlane, srcDepth, srcHeight, srcWidth};
    dst_tz = {nbatch, nInputPlane, dstDepth, dstHeight, dstWidth};
    kernel = {kD, kH, kW};
    stride = {dD, dH, dW};
    padding = {padD, padH, padW};
  }

  auto format_any = memory::format_tag::any;
  auto src_md = memory::desc(src_tz, data_t, format);
  auto idx_md = memory::desc(dst_tz, data_t, format);
  auto dst_md = memory::desc(dst_tz, data_t, format);

  auto dst_md_any = memory::desc(dst_tz, data_t, format_any);

  if (Settings::I().is_onednn_layout_enabled()) {
    auto src_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src);
    src_md = src_ctx.is_plain() ? src_md : src_ctx.meta();
  }

#ifdef USE_PRIMITIVE_CACHE
    lru_key_t key;
    create_key(key, src_md, dst_md_any, stride, kernel, padding, padding, alg_kind);
#endif
    auto pooling_fwd_desc = pooling_forward::desc(
        prop_kind,
        alg_kind,
        src_md,
        dst_md_any,
        stride,
        kernel,
        padding,
        padding);

  auto pooling_fwd_pd = pooling_forward::primitive_desc(pooling_fwd_desc, engine);

  auto expected_dst_md = pooling_fwd_pd.dst_desc();

  memory src_usr_m, dst_usr_m;
  if (!Settings::I().is_onednn_layout_enabled() ||
      src.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      src.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    src_usr_m = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
    dst_usr_m = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());
  } else {
    src_usr_m = dpcpp_onednn_memory(src_md, engine, src.data_ptr());

    if (expected_dst_md != dst_md) {
      // reallocate memory due to padding needed by oneDNN in some blk fmt
      if (src.is_quantized()) {
        auto quantizer = dpcpp_make_per_tensor_affine_quantizer(
            src.q_scale(),
            src.q_zero_point(),
            at::typeMetaToScalarType(src.options().dtype()));
        dst = empty_opaque_qtensor(expected_dst_md, c10::nullopt, quantizer);
      } else {
        dst = empty_opaque_tensor(expected_dst_md, src.options(), c10::nullopt);
      }
      dst_usr_m = dpcpp_onednn_memory(expected_dst_md, engine, dst.data_ptr());
    } else {
      dst_usr_m = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());
    }
  }

  auto src_m = src_usr_m;
  auto dst_m = dst_usr_m;

  if (prop_kind == dnnl::prop_kind::forward_training) {
    at::Tensor idx_;
    memory idx_m;
    if (!Settings::I().is_onednn_layout_enabled() ||
        src.is_contiguous(at::MemoryFormat::ChannelsLast) ||
        src.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
      idx_ = at::empty({dst_tz}, at::TensorOptions(at::kXPU).dtype(at::kInt));
      idx_m = dpcpp_onednn_memory(idx_md, engine, idx_.data_ptr());
    } else {
      auto expected_idx_md = pooling_fwd_pd.workspace_desc();
      idx_ = empty_opaque_tensor(
          expected_idx_md, at::TensorOptions(at::kXPU).dtype(at::kInt), c10::nullopt);
      idx_m = dpcpp_onednn_memory(expected_idx_md, engine, idx_.data_ptr());
    }

#ifdef USE_PRIMITIVE_CACHE
    auto pooling_fwd = fetch_or_create_m<pooling_forward>(key, pooling_fwd_pd);
#else
    auto pooling_fwd = pooling_forward(pooling_fwd_pd);
#endif

    DPCPP_ONEDNN_EXEC(
        pooling_fwd,
        strm,
        {{DNNL_ARG_SRC, src_m},
         {DNNL_ARG_DST, dst_m},
         {DNNL_ARG_WORKSPACE, idx_m}});

    if (!Settings::I().is_onednn_layout_enabled() ||
        src.is_contiguous(at::MemoryFormat::ChannelsLast) ||
        src.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
      dtype_convert_by_scalar(idx.data_ptr<int64_t>(), idx_.data_ptr<int32_t>(), idx_.numel());
    } else {
      // reorder if materialized
      auto idx_internal_ctx = DPCPPTensorContext::release_tensor_ctx(idx_);
      DPCPPTensorContext::set_tensor_ctx(idx, std::move(idx_internal_ctx));
    }
  } else {
    idx = at::empty({dst_tz}, at::TensorOptions(at::kXPU).dtype(at::kInt));
#ifdef USE_PRIMITIVE_CACHE
    auto pooling_fwd = fetch_or_create_m<pooling_forward>(key, pooling_fwd_pd);
#else
    auto pooling_fwd = pooling_forward(pooling_fwd_pd);
#endif
    DPCPP_ONEDNN_EXEC(
        pooling_fwd,
        strm,
        {{DNNL_ARG_SRC, src_m}, {DNNL_ARG_DST, dst_m}});
  }

  return {dst, idx};
}

template <alg alg_kind>
static at::Tensor pooling_backward(
    at::Tensor& diff_src,
    const at::Tensor& diff_dst,
    int64_t nbatch,
    int64_t nInputPlane,
    int64_t diff_src_depth,
    int64_t diff_src_height,
    int64_t diff_src_width,
    int64_t diff_dst_depth,
    int64_t diff_dst_height,
    int64_t diff_dst_width,
    int kD,
    int kH,
    int kW,
    int dD,
    int dH,
    int dW,
    int padD,
    int padH,
    int padW) {
  at::Device curDevice = at::Device(at::kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();
  prop_kind prop_kind = dnnl::prop_kind::forward_training;

  auto data_t = get_onednn_dtype(diff_dst);
  if (data_t == memory::data_type::f16) {
    // rise error
  }

  memory::format_tag format;

  memory::dims diff_src_tz;
  memory::dims diff_dst_tz;
  memory::dims kernel;
  memory::dims stride;
  memory::dims padding;
  memory::format_tag format_any = memory::format_tag::any;

  if (diff_src_depth == 0) {
    format = diff_src.is_contiguous(at::MemoryFormat::ChannelsLast) ?
             memory::format_tag::nhwc :
             memory::format_tag::nchw;

    diff_src_tz = {nbatch, nInputPlane, diff_src_height, diff_src_width};
    diff_dst_tz = {nbatch, nInputPlane, diff_dst_height, diff_dst_width};
    kernel = {kH, kW};
    stride = {dH, dW};
    padding = {padH, padW};
  } else {
    format = diff_src.is_contiguous(at::MemoryFormat::ChannelsLast3d) ?
             memory::format_tag::ndhwc :
             memory::format_tag::ncdhw;

    diff_src_tz = {nbatch,
                   nInputPlane,
                   diff_src_depth,
                   diff_src_height,
                   diff_src_width};
    diff_dst_tz = {nbatch,
                   nInputPlane,
                   diff_dst_depth,
                   diff_dst_height,
                   diff_dst_width};
    kernel = {kD, kH, kW};
    stride = {dD, dH, dW};
    padding = {padD, padH, padW};
  }

  auto diff_src_md = memory::desc({diff_src_tz}, data_t, format);
  auto diff_src_md_any = memory::desc({diff_src_tz}, data_t, format_any);
  auto diff_dst_md = memory::desc({diff_dst_tz}, data_t, format);

  if (Settings::I().is_onednn_layout_enabled()) {
    auto diff_dst_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(diff_dst);
    diff_dst_md = diff_dst_ctx.is_plain()? diff_dst_md : diff_dst_ctx.meta();
  }

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, diff_src_md_any, diff_dst_md, stride, kernel, padding, padding, alg_kind);
#endif
  auto pooling_fwd_desc = pooling_forward::desc(
      prop_kind, alg_kind, diff_src_md_any, diff_dst_md, stride, kernel, padding, padding);

  auto pooling_fwd_pd = pooling_forward::primitive_desc(pooling_fwd_desc, engine);
  auto pooling_bwd_desc = dnnl::pooling_backward::desc(
      alg_kind, diff_src_md_any, diff_dst_md, stride, kernel, padding, padding);

  auto pooling_bwd_pd = dnnl::pooling_backward::primitive_desc(pooling_bwd_desc, engine, pooling_fwd_pd);

#ifdef USE_PRIMITIVE_CACHE
  auto pooling_bwd = fetch_or_create_m<dnnl::pooling_backward>(key, pooling_bwd_pd);
#else
  auto pooling_bwd = dnnl::pooling_backward(pooling_bwd_pd);
#endif

  memory diff_src_m, diff_dst_m;
  if (!Settings::I().is_onednn_layout_enabled()
      || diff_src.is_contiguous(at::MemoryFormat::ChannelsLast)
      || diff_src.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    diff_dst_m = dpcpp_onednn_memory(diff_dst_md, engine, diff_dst.data_ptr());

    diff_src_m = dpcpp_onednn_memory(diff_src_md, engine, diff_src.data_ptr());
  } else {
    diff_dst_m = dpcpp_onednn_memory(diff_dst_md, engine, diff_dst.data_ptr());

    auto plain_diff_src_md = diff_src_md;
    auto expected_diff_src_md = pooling_bwd_pd.diff_src_desc();
    if (expected_diff_src_md != plain_diff_src_md) {
      diff_src = empty_opaque_tensor(expected_diff_src_md, diff_dst.options(), c10::nullopt);
      diff_src_m = dpcpp_onednn_memory(expected_diff_src_md, engine, diff_src.data_ptr());
    } else {
      diff_src_m = dpcpp_onednn_memory(plain_diff_src_md, engine, diff_src.data_ptr());
    }
  }

  DPCPP_ONEDNN_EXEC(
      pooling_bwd,
      strm,
      {{DNNL_ARG_DIFF_DST, diff_dst_m},
       {DNNL_ARG_DIFF_SRC, diff_src_m}});

  return diff_src;
}

template <alg alg_kind>
static at::Tensor pooling_backward(
    at::Tensor& diff_src,
    const at::Tensor& diff_dst,
    const at::Tensor& idx,
    int64_t nbatch,
    int64_t nInputPlane,
    int64_t diff_src_depth,
    int64_t diff_src_height,
    int64_t diff_src_width,
    int64_t diff_dst_depth,
    int64_t diff_dst_height,
    int64_t diff_dst_width,
    int kD,
    int kH,
    int kW,
    int dD,
    int dH,
    int dW,
    int padD,
    int padH,
    int padW) {
  at::Device curDevice = at::Device(at::kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  auto prop_kind = dnnl::prop_kind::forward_training;
  auto data_t = get_onednn_dtype(diff_dst);
  if (data_t == memory::data_type::f16) {
    // rise error
  }

  memory::format_tag format;
  memory::dims diff_src_tz;
  memory::dims diff_dst_tz;
  memory::dims kernel;
  memory::dims stride;
  memory::dims padding;

  if (diff_src_depth == 0) {
    format = diff_src.is_contiguous(at::MemoryFormat::ChannelsLast) ?
             memory::format_tag::nhwc :
             memory::format_tag::nchw;

    diff_src_tz = {nbatch, nInputPlane, diff_src_height, diff_src_width};
    diff_dst_tz = {nbatch, nInputPlane, diff_dst_height, diff_dst_width};
    kernel = {kH, kW};
    stride = {dH, dW};
    padding = {padH, padW};
  } else {
    format = diff_src.is_contiguous(at::MemoryFormat::ChannelsLast3d) ?
             memory::format_tag::ndhwc :
             memory::format_tag::ncdhw;
    diff_src_tz = {nbatch, nInputPlane, diff_src_depth, diff_src_height, diff_src_width};
    diff_dst_tz = {nbatch, nInputPlane, diff_dst_depth, diff_dst_height, diff_dst_width};
    kernel = {kD, kH, kW};
    stride = {dD, dH, dW};
    padding = {padD, padH, padW};
  }

  auto format_any = memory::format_tag::any;
  auto diff_src_md = memory::desc({diff_src_tz}, data_t, format);
  auto diff_dst_md = memory::desc({diff_dst_tz}, data_t, format);
  auto diff_src_md_any = memory::desc({diff_src_tz}, data_t, format_any);
  if (Settings::I().is_onednn_layout_enabled()) {
    auto diff_dst_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(diff_dst);
    diff_dst_md = diff_dst_ctx.is_plain() ? diff_dst_md : diff_dst_ctx.meta();
  }

#ifdef USE_PRIMITIVE_CACHE
    lru_key_t key;
    create_key(key, diff_src_md_any, diff_dst_md, stride, kernel, padding, padding, alg_kind);
#endif
  auto pooling_fwd_desc = pooling_forward::desc(
      prop_kind, alg_kind, diff_src_md_any, diff_dst_md,
      stride, kernel, padding, padding);
  auto pooling_bwd_desc = dnnl::pooling_backward::desc(
      alg_kind, diff_src_md_any, diff_dst_md, stride, kernel, padding, padding);

  auto pooling_fwd_pd = pooling_forward::primitive_desc(pooling_fwd_desc, engine);
  auto pooling_bwd_pd = dnnl::pooling_backward::primitive_desc(
      pooling_bwd_desc, engine, pooling_fwd_pd);

  auto expected_diff_src_md = pooling_bwd_pd.diff_src_desc();
  memory diff_src_usr_m, diff_dst_usr_m, idx_usr_m;
  if (!Settings::I().is_onednn_layout_enabled()
      || diff_src.is_contiguous(at::MemoryFormat::ChannelsLast)
      || diff_src.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    diff_dst_usr_m = dpcpp_onednn_memory(
        {diff_dst_tz, data_t, format}, engine, diff_dst.data_ptr());

    diff_src_usr_m = dpcpp_onednn_memory(
        {diff_src_tz, data_t, format}, engine, diff_src.data_ptr());
  } else {
    diff_dst_usr_m = dpcpp_onednn_memory(
        {diff_dst_tz, data_t, format}, engine, diff_dst.data_ptr());
    auto plain_diff_src_md = diff_src_md;

    if (expected_diff_src_md != plain_diff_src_md) {
      diff_src = empty_opaque_tensor(expected_diff_src_md, diff_dst.options(), c10::nullopt);
      diff_src_usr_m = dpcpp_onednn_memory(expected_diff_src_md, engine, diff_src.data_ptr());
    } else {
      diff_src_usr_m = dpcpp_onednn_memory(diff_src_md, engine, diff_src.data_ptr());
    }
  }

  auto diff_dst_m = diff_dst_usr_m;
  auto diff_src_m = diff_src_usr_m;

  auto expexted_idx_md = pooling_bwd_pd.workspace_desc();
  auto idx_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(idx);
  at::Tensor idx_usr = idx;
  if (idx_ctx.is_plain()) {
    idx_usr = at::empty({diff_dst_tz}, at::TensorOptions(at::kXPU).dtype(at::kInt));
    dtype_convert_by_scalar(idx_usr.data_ptr<int32_t>(), idx.data_ptr<int64_t>(), idx_usr.numel());

    idx_usr_m = dpcpp_onednn_memory(
        {diff_dst_tz, (memory::data_type)expexted_idx_md.data.data_type, format},
        engine, idx_usr.data_ptr());
  } else {
    idx_usr_m = dpcpp_onednn_memory(idx_ctx.meta(), engine, idx.data_ptr());
  }

  at::Tensor idx_opt;
  auto idx_m = idx_usr_m;
  if (Settings::I().is_onednn_layout_enabled()) {
    if (idx_usr_m.get_desc() != expexted_idx_md) {
      idx_opt = empty_opaque_tensor(
          expexted_idx_md, at::TensorOptions(at::kXPU).dtype(at::kInt), c10::nullopt);
      idx_m = dpcpp_onednn_memory(expexted_idx_md, engine, idx_opt.data_ptr());
      xpu::oneDNN::reorder(idx_usr, idx_opt);
    }
  }

  auto pooling_bwd = dnnl::pooling_backward(pooling_bwd_pd);
  DPCPP_ONEDNN_EXEC(
      pooling_bwd,
      strm,
      {{DNNL_ARG_DIFF_DST, diff_dst_m},
       {DNNL_ARG_DIFF_SRC, diff_src_m},
       {DNNL_ARG_WORKSPACE, idx_m}});

  return diff_src;
}

}}
