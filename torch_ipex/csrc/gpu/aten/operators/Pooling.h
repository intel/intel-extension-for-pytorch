#pragma once

#include <ATen/ipex_type_dpcpp_customized.h>
#include <core/Quantizer.h>
#include <core/Memory.h>
#include <core/Runtime.h>
#include <tensor/Context.h>
#include <utils/Env.h>

#ifdef USE_PRIMITIVE_CACHE
#include <oneDNN/LRUCache.h>
#endif

using namespace dnnl;
using namespace at::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <algorithm alg_kind>
static void avg_pool_out_frame(
    const Tensor& src,
    Tensor& dst,
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
  Device curDevice = Device(kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  prop_kind prop_kind = dnnl::prop_kind::forward_training;
  auto data_t = dt_to_dnnl(src.scalar_type());
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
    format = memory::format_tag::nchw;

    src_tz = {nbatch, nInputPlane, srcHeight, srcWidth};
    dst_tz = {nbatch, nInputPlane, dstHeight, dstWidth};
    kernel = {kH, kW};
    stride = {dH, dW};
    padding = {padH, padW};
  } else {
    format = memory::format_tag::ncdhw;

    src_tz = {nbatch, nInputPlane, srcDepth, srcHeight, srcWidth};
    dst_tz = {nbatch, nInputPlane, dstDepth, dstHeight, dstWidth};
    kernel = {kD, kH, kW};
    stride = {dD, dH, dW};
    padding = {padD, padH, padW};
  }

  auto src_md = memory::desc({src_tz}, data_t, format);
  auto dst_md = memory::desc({dst_tz}, data_t, format);
  auto dst_md_any = memory::desc({dst_tz}, data_t, format_any);

  if (lazy_reorder_enabled()) {
    auto src_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src);
    src_md = src_ctx.is_plain() ? memory::desc({src_tz}, data_t, format)
                                    : src_ctx.meta();
  }

  lru_key_t key;
#ifdef USE_PRIMITIVE_CACHE
  create_key(key, src_md, dst_md_any, stride, kernel, padding, padding, alg_kind);
#endif
  auto pooling_forward_desc = pooling_forward::desc(
      prop_kind,
      alg_kind,
      src_md,
      dst_md_any,
      stride,
      kernel,
      padding,
      padding);

  auto pooling_forward_pd = pooling_forward::primitive_desc(pooling_forward_desc, engine);

  memory src_memory, dst_memory;
  if (!lazy_reorder_enabled()) {
    src_memory = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
    dst_memory = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());
  } else {
    src_memory = dpcpp_onednn_memory(src_md, engine, src.data_ptr());

    auto plain_dst_md = dst_md;
    auto expected_dst_md = pooling_forward_pd.dst_desc();
    if (expected_dst_md != plain_dst_md) {
      // reallocate memory due to padding needed by oneDNN in some blk fmt
      if (src.is_quantized()) {
        auto quantizer = at::dpcpp::make_per_tensor_affine_quantizer(
            src.q_scale(),
            src.q_zero_point(),
            typeMetaToScalarType(src.options().dtype()));
        dst = empty_opaque_qtensor(expected_dst_md, c10::nullopt, quantizer);
      } else {
        dst = empty_opaque_tensor(expected_dst_md, src.options(), c10::nullopt);
      }
      dst_memory = dpcpp_onednn_memory(expected_dst_md, engine, dst.data_ptr());
    } else {
      dst_memory = dpcpp_onednn_memory(plain_dst_md, engine, dst.data_ptr());
    }
  }

#ifdef USE_PRIMITIVE_CACHE
  auto pool_forward = fetch_or_create_m<pooling_forward>(key, pooling_forward_pd);
#else
  auto pool_forward = pooling_forward(pooling_forward_pd);
#endif
  DPCPP_ONEDNN_EXEC(
      pool_forward,
      strm,
      {{DNNL_ARG_SRC, src_memory}, {DNNL_ARG_DST, dst_memory}});
}

template <algorithm alg_kind>
static void avg_pool_backward_out_frame(
    Tensor& diff_src,
    const Tensor& diff_dst,
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
  at::Device curDevice = at::Device(kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();
  prop_kind prop_kind = dnnl::prop_kind::forward_training;

  auto data_t = dt_to_dnnl(diff_dst.scalar_type());
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
    format = memory::format_tag::nchw;

    diff_src_tz = {nbatch, nInputPlane, diff_src_height, diff_src_width};
    diff_dst_tz = {nbatch, nInputPlane, diff_dst_height, diff_dst_width};
    kernel = {kH, kW};
    stride = {dH, dW};
    padding = {padH, padW};
  } else {
    format = memory::format_tag::ncdhw;

    diff_src_tz = {nbatch, nInputPlane, diff_src_depth, diff_src_height, diff_src_width};
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

  if (lazy_reorder_enabled()) {
    auto diff_dst_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(diff_dst);
    diff_dst_md = diff_dst_ctx.is_plain()? diff_dst_md : diff_dst_ctx.meta();
  }

  lru_key_t key;
#ifdef USE_PRIMITIVE_CACHE
  create_key(key, diff_src_md_any, diff_dst_md, stride, kernel, padding, padding, alg_kind);
#endif
  auto pooling_forward_desc = pooling_forward::desc(
      prop_kind, alg_kind, diff_src_md_any, diff_dst_md, stride, kernel, padding, padding);

  auto pooling_forward_pd = pooling_forward::primitive_desc(pooling_forward_desc, engine);
  auto pooling_backward_desc = pooling_backward::desc(
      alg_kind, diff_src_md_any, diff_dst_md, stride, kernel, padding, padding);

  auto pooling_backward_pd = pooling_backward::primitive_desc(pooling_backward_desc, engine, pooling_forward_pd);

#ifdef USE_PRIMITIVE_CACHE
  auto pool_backward = fetch_or_create_m<pooling_backward>(key, pooling_backward_pd);
#else
  auto pool_backward = pooling_backward(pooling_backward_pd);
#endif

  memory diff_src_memory, diff_dst_memory;
  if (!lazy_reorder_enabled()) {
    diff_dst_memory = dpcpp_onednn_memory(diff_dst_md, engine, diff_dst.data_ptr());

    diff_src_memory = dpcpp_onednn_memory(diff_src_md, engine, diff_src.data_ptr());
  } else {
    diff_dst_memory = dpcpp_onednn_memory(diff_dst_md, engine, diff_dst.data_ptr());

    auto plain_diff_src_md = diff_src_md;
    auto expected_diff_src_md = pooling_backward_pd.diff_src_desc();
    if (expected_diff_src_md != plain_diff_src_md) {
      diff_src = empty_opaque_tensor(expected_diff_src_md, diff_dst.options(), c10::nullopt);
      diff_src_memory = dpcpp_onednn_memory(expected_diff_src_md, engine, diff_src.data_ptr());
    } else {
      diff_src_memory = dpcpp_onednn_memory(plain_diff_src_md, engine, diff_src.data_ptr());
    }
  }

  DPCPP_ONEDNN_EXEC(
      pool_backward,
      strm,
      {{DNNL_ARG_DIFF_DST, diff_dst_memory},
       {DNNL_ARG_DIFF_SRC, diff_src_memory}});
}

template <algorithm alg_kind>
static void max_pool_out_frame(
    const Tensor& src,
    Tensor& dst,
    Tensor& indices,
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
  at::Device curDevice = at::Device(kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  auto prop_kind = dnnl::prop_kind::forward_training;
  auto data_t = dt_to_dnnl(src.scalar_type());
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
    format = memory::format_tag::nchw;
    src_tz = {nbatch, nInputPlane, srcHeight, srcWidth};
    dst_tz = {nbatch, nInputPlane, dstHeight, dstWidth};
    kernel = {kH, kW};
    stride = {dH, dW};
    padding = {padH, padW};
  } else {
    format = memory::format_tag::ncdhw;
    src_tz = {nbatch, nInputPlane, srcDepth, srcHeight, srcWidth};
    dst_tz = {nbatch, nInputPlane, dstDepth, dstHeight, dstWidth};
    kernel = {kD, kH, kW};
    stride = {dD, dH, dW};
    padding = {padD, padH, padW};
  }

  auto format_any = memory::format_tag::any;
  auto src_md = memory::desc(src_tz, data_t, format);
  auto indices_md = memory::desc(dst_tz, data_t, format);
  auto dst_md = memory::desc(dst_tz, data_t, format);

  auto dst_md_any = memory::desc(dst_tz, data_t, format_any);

  if (lazy_reorder_enabled()) {
    auto src_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src);
    src_md = src_ctx.is_plain() ? src_md : src_ctx.meta();
  }

  lru_key_t key;
#ifdef USE_PRIMITIVE_CACHE
    create_key(key, src_md, dst_md_any, stride, kernel, padding, padding, alg_kind);
#endif
    auto pooling_forward_desc = pooling_forward::desc(
        prop_kind,
        alg_kind,
        src_md,
        dst_md_any,
        stride,
        kernel,
        padding,
        padding);

  auto pooling_forward_pd = pooling_forward::primitive_desc(pooling_forward_desc, engine);

  auto expected_dst_md = pooling_forward_pd.dst_desc();

  memory src_usr_memory, dst_usr_memory;
  if (!lazy_reorder_enabled()) {
    src_usr_memory = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
    dst_usr_memory = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());
  } else {
    src_usr_memory = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
    auto plain_dst_md = dst_md;

    if (expected_dst_md != plain_dst_md) {
      // reallocate memory due to padding needed by oneDNN in some blk fmt
      if (src.is_quantized()) {
        auto quantizer = at::dpcpp::make_per_tensor_affine_quantizer(
            src.q_scale(),
            src.q_zero_point(),
            typeMetaToScalarType(src.options().dtype()));
        dst = empty_opaque_qtensor(expected_dst_md, c10::nullopt, quantizer);
      } else {
        dst = empty_opaque_tensor(expected_dst_md, src.options(), c10::nullopt);
      }
      dst_usr_memory = dpcpp_onednn_memory(expected_dst_md, engine, dst.data_ptr());
    } else {
      dst_usr_memory = dpcpp_onednn_memory(
          dst_md, engine, dst.data_ptr());
    }
  }

  auto src_memory = src_usr_memory;
  auto dst_memory = dst_usr_memory;

  if (prop_kind == dnnl::prop_kind::forward_training) {
    Tensor indices_;
    memory indices_usr_memory;
    if (!lazy_reorder_enabled()) {
      indices_ = at::empty({dst_tz}, at::TensorOptions(kXPU).dtype(kInt));
      indices_usr_memory = dpcpp_onednn_memory(indices_md, engine, indices_.data_ptr());
    } else {
      auto expected_indices_md = pooling_forward_pd.workspace_desc();
      indices_ = empty_opaque_tensor(
          expected_indices_md, at::TensorOptions(kXPU).dtype(kInt), c10::nullopt);
      indices_usr_memory = dpcpp_onednn_memory(expected_indices_md, engine, indices_.data_ptr());
    }
    auto indices_memory = indices_usr_memory;

#ifdef USE_PRIMITIVE_CACHE
    auto pool_forward = fetch_or_create_m<pooling_forward>(key, pooling_forward_pd);
#else
    auto pool_forward = pooling_forward(pooling_forward_pd);
#endif
    DPCPP_ONEDNN_EXEC(
        pool_forward,
        strm,
        {{DNNL_ARG_SRC, src_memory},
         {DNNL_ARG_DST, dst_memory},
         {DNNL_ARG_WORKSPACE, indices_memory}});

    if (!lazy_reorder_enabled()) {
      dpcppMemoryCopyType(indices.data_ptr<int64_t>(),indices_.data_ptr<int32_t>(),indices_.numel());
    } else {
      // reorder if materialized
      auto indices_internal_ctx =DPCPPTensorContext::release_tensor_ctx(indices_);
      DPCPPTensorContext::set_tensor_ctx(indices, std::move(indices_internal_ctx));
    }
  } else {
    indices = at::empty({dst_tz}, at::TensorOptions(kXPU).dtype(kInt));
#ifdef USE_PRIMITIVE_CACHE
    auto pool_forward = fetch_or_create_m<pooling_forward>(key, pooling_forward_pd);
#else
    auto pool_forward = pooling_forward(pooling_forward_pd);
#endif
    DPCPP_ONEDNN_EXEC(
        pool_forward,
        strm,
        {{DNNL_ARG_SRC, src_memory}, {DNNL_ARG_DST, dst_memory}});
  }
}

template <algorithm alg_kind>
static void max_pool_backward_out_frame(
    Tensor& diff_src,
    const Tensor& diff_dst,
    const Tensor& indices,
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
  auto data_t = dt_to_dnnl(diff_dst.scalar_type());
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
    format = memory::format_tag::nchw;
    diff_src_tz = {nbatch, nInputPlane, diff_src_height, diff_src_width};
    diff_dst_tz = {nbatch, nInputPlane, diff_dst_height, diff_dst_width};
    kernel = {kH, kW};
    stride = {dH, dW};
    padding = {padH, padW};
  } else {
    format = memory::format_tag::ncdhw;
    diff_src_tz = {nbatch, nInputPlane, diff_src_depth, diff_src_height, diff_src_width};
    diff_dst_tz = {nbatch, nInputPlane, diff_dst_depth, diff_dst_height, diff_dst_width};
    kernel = {kD, kH, kW};
    stride = {dD, dH, dW};
    padding = {padD, padH, padW};
  }

  auto format_any = memory::format_tag::any;
  auto diff_src_md = memory::desc({diff_src_tz}, data_t, format);
  auto diff_dst_md = memory::desc({diff_dst_tz}, data_t, format);
  auto diff_src_md_any = memory::desc({diff_src_tz}, data_t, format);
  if (lazy_reorder_enabled()) {
    auto diff_dst_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(diff_dst);
    diff_dst_md = diff_dst_ctx.is_plain() ? diff_dst_md : diff_dst_ctx.meta();
  }

  lru_key_t key;
#ifdef USE_PRIMITIVE_CACHE
    create_key(key, diff_src_md_any, diff_dst_md, stride, kernel, padding, padding, alg_kind);
#endif
  auto pooling_forward_desc = pooling_forward::desc(
      prop_kind, alg_kind, diff_src_md_any, diff_dst_md,
      stride, kernel, padding, padding);
  auto pooling_backward_desc = pooling_backward::desc(
      alg_kind, diff_src_md_any, diff_dst_md, stride, kernel, padding, padding);


  auto pooling_forward_pd = pooling_forward::primitive_desc(pooling_forward_desc, engine);
  auto pooling_backward_pd = pooling_backward::primitive_desc(
      pooling_backward_desc, engine, pooling_forward_pd);

  auto expected_diff_src_md = pooling_backward_pd.diff_src_desc();
  memory diff_src_usr_memory, diff_dst_usr_memory, indices_usr_memory;
  if (!lazy_reorder_enabled()) {
    diff_dst_usr_memory = dpcpp_onednn_memory(
        {diff_dst_tz, data_t, format}, engine, diff_dst.data_ptr());

    diff_src_usr_memory = dpcpp_onednn_memory(
        {diff_src_tz, data_t, format}, engine, diff_src.data_ptr());
  } else {
    diff_dst_usr_memory = dpcpp_onednn_memory(
        {diff_dst_tz, data_t, format}, engine, diff_dst.data_ptr());
    auto plain_diff_src_md = diff_src_md;

    if (expected_diff_src_md != plain_diff_src_md) {
      diff_src = empty_opaque_tensor(expected_diff_src_md, diff_dst.options(), c10::nullopt);
      diff_src_usr_memory = dpcpp_onednn_memory(expected_diff_src_md, engine, diff_src.data_ptr());
    } else {
      diff_src_usr_memory = dpcpp_onednn_memory(diff_src_md, engine, diff_src.data_ptr());
    }
  }

  auto diff_dst_memory = diff_dst_usr_memory;
  auto diff_src_memory = diff_src_usr_memory;

  auto expexted_indices_md = pooling_backward_pd.workspace_desc();
  auto indices_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(indices);
  Tensor indices_usr;
  if (indices_ctx.is_plain()) {
    indices_usr = at::empty({diff_dst_tz}, at::TensorOptions(kXPU).dtype(kInt));
    dpcppMemoryCopyType(indices_usr.data_ptr<int32_t>(), indices.data_ptr<int64_t>(), indices_usr.numel());

    indices_usr_memory = dpcpp_onednn_memory(
        {diff_dst_tz, (memory::data_type)expexted_indices_md.data.data_type, format},
        engine, indices_usr.data_ptr());
  } else {
    indices_usr_memory = dpcpp_onednn_memory(indices_ctx.meta(), engine, indices.data_ptr());
  }

  Tensor indices_opt;
  auto indices_memory = indices_usr_memory;
  if (lazy_reorder_enabled()) {
    if (indices_usr_memory.get_desc() != expexted_indices_md) {
      indices_opt = at::empty({expexted_indices_md.get_size() / indices.itemsize()},
        at::TensorOptions(kXPU).dtype(kInt));
      indices_memory = dpcpp_onednn_memory(expexted_indices_md, engine, indices_opt.data_ptr());
      DPCPP_ONEDNN_EXEC(reorder(indices_usr_memory, indices_memory),
          strm, {{DNNL_ARG_FROM, indices_usr_memory}, {DNNL_ARG_TO, indices_memory}});
    }
  }

  auto pool_backward = pooling_backward(pooling_backward_pd);
  DPCPP_ONEDNN_EXEC(
      pool_backward,
      strm,
      {{DNNL_ARG_DIFF_DST, diff_dst_memory},
       {DNNL_ARG_DIFF_SRC, diff_src_memory},
       {DNNL_ARG_WORKSPACE, indices_memory}});
}

} // namespace impl
} // namespace AtenIpexTypeXPU
} // namespace at
