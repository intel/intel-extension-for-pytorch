#pragma once
#include <ATen/ATen.h>
#include <ATen/Config.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>
#include <tensor/Context.h>
#include <ATen/ipex_type_dpcpp_customized.h>

#ifdef USE_PRIMITIVE_CACHE
#include <oneDNN/LRUCache.h>
#endif

using namespace dnnl;
using dnnl::algorithm;
using namespace at::dpcpp;

namespace at {
namespace dpcpp {
namespace oneDNN {

template <algorithm alg_kind>
static inline void eltwise(
    at::Tensor& dst,
    const at::Tensor& src,
    float alpha = 0,
    float beta = 0) {
  Device curDevice = Device(kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  std::vector<int64_t> dims;
  for (size_t i = 0; i < src.dim(); i++) {
    dims.push_back(src.size(i));
  }

  memory::dims src_tz = dims;
  auto data_t = dt_to_dnnl(src.scalar_type());
  auto format_data = get_dnnl_default_format(src.dim());
  auto src_md = memory::desc({src_tz}, data_t, format_data);

  memory src_usr_memory;
  if (!lazy_reorder_enabled()) {
    src_usr_memory = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
  } else {
    auto src_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src);
    src_md = src_ctx.is_plain() ? src_md : src_ctx.meta();
    src_usr_memory = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
  }


#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, alg_kind, src_md, alpha, beta);
#endif

  eltwise_forward::desc eltwise_eltwiseFwd_desc(prop_kind::forward, alg_kind, src_md, alpha, beta);
  auto eltwise_forward_pd = eltwise_forward::primitive_desc(eltwise_eltwiseFwd_desc, engine);

  memory dst_usr_memory;
  if (!lazy_reorder_enabled()) {
    if (!dst.defined())
      dst = at::empty_like(src);
    dst_usr_memory = dpcpp_onednn_memory(eltwise_forward_pd.dst_desc(), engine, dst.data_ptr());
  } else {
    if (dst.defined()) {
      auto dst_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(dst);
      auto dst_md = dst_ctx.is_plain() ? src_md : dst_ctx.meta();
      dst_usr_memory = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());
    } else {
      auto plain_dst_md = memory::desc({src_tz}, data_t, format_data);
      auto expected_dst_md = eltwise_forward_pd.dst_desc();
      if (plain_dst_md != expected_dst_md) {
        dst = at::AtenIpexTypeXPU::empty_opaque_tensor(
            expected_dst_md, src.options(), c10::nullopt);
        dst_usr_memory = dpcpp_onednn_memory(expected_dst_md, engine, dst.data_ptr());
      } else {
        dst = at::empty_like(src);
        dst_usr_memory = dpcpp_onednn_memory(plain_dst_md, engine, dst.data_ptr());
      }
    }
  }

  auto strm = GpuStreamManager::Instance().get_stream();

#ifdef USE_PRIMITIVE_CACHE
  auto eltwise_fwd =
      fetch_or_create_m<dnnl::eltwise_forward>(key, eltwise_forward_pd);
#else
  auto eltwise_fwd = dnnl::eltwise_forward(eltwise_forward_pd);
#endif

  DPCPP_ONEDNN_EXEC(
      eltwise_fwd,
      strm,
      {{DNNL_ARG_SRC, src_usr_memory},
       {DNNL_ARG_DST, dst_usr_memory}});
}

template <algorithm alg_kind>
static inline void eltwise_backward(
    at::Tensor& diff_src,
    const at::Tensor& src_dst,
    const at::Tensor& diff_dst,
    float alpha = 0,
    float beta = 0) {
  Device curDevice = Device(kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  auto data_t = dt_to_dnnl(src_dst.scalar_type());
  std::vector<int64_t> src_dst_dims;
  for (size_t i = 0; i < src_dst.dim(); i++) {
    src_dst_dims.push_back(src_dst.size(i));
  }

  memory::dims src_dst_tz = src_dst_dims;
  auto format_data = get_dnnl_default_format(src_dst.dim());
  auto src_dst_md = memory::desc({src_dst_tz}, data_t, format_data);
  auto diff_dst_md = memory::desc({src_dst_tz}, data_t, format_data);

  memory src_dst_usr_memory, diff_dst_usr_memory;
  if (!lazy_reorder_enabled()) {
    src_dst_usr_memory  = dpcpp_onednn_memory(src_dst_md, engine, src_dst.data_ptr());
    diff_dst_usr_memory = dpcpp_onednn_memory(diff_dst_md, engine, diff_dst.data_ptr());
  } else {
    auto src_dst_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src_dst);
    src_dst_md = src_dst_ctx.is_plain() ? src_dst_md : src_dst_ctx.meta();
    src_dst_usr_memory = dpcpp_onednn_memory(src_dst_md, engine, src_dst.data_ptr());

    auto diff_dst_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(diff_dst);
    diff_dst_md = diff_dst_ctx.is_plain() ? diff_dst_md : diff_dst_ctx.meta();
    diff_dst_usr_memory = dpcpp_onednn_memory(diff_dst_md, engine, diff_dst.data_ptr());
  }

  eltwise_forward::desc eltwise_eltwiseFwd_desc(
      prop_kind::forward_training, alg_kind, src_dst_md, alpha, beta);
  auto eltwise_forward_pd =eltwise_forward::primitive_desc(eltwise_eltwiseFwd_desc, engine);
  eltwise_backward::desc eltwise_eltwiseBwd_desc(alg_kind, diff_dst_md, src_dst_md, alpha, beta);
  auto eltwise_backward_pd = eltwise_backward::primitive_desc(
      eltwise_eltwiseBwd_desc, engine, eltwise_forward_pd);

  memory diff_src_usr_memory;
  if (!lazy_reorder_enabled()) {
    if (!diff_src.defined()) {
      diff_src = at::empty_like(src_dst);
    }
    auto diff_src_md = memory::desc({src_dst_tz, data_t, format_data});
    diff_src_usr_memory = dpcpp_onednn_memory(diff_src_md, engine, diff_src.data_ptr());
  } else {
    auto plain_diff_src_md = memory::desc({src_dst_tz}, data_t, format_data);
    auto expected_diff_src_md = eltwise_backward_pd.diff_src_desc();
    if (plain_diff_src_md != expected_diff_src_md) {
      diff_src = at::AtenIpexTypeXPU::empty_opaque_tensor(
            expected_diff_src_md, src_dst.options(), c10::nullopt);
      diff_src_usr_memory = dpcpp_onednn_memory(expected_diff_src_md, engine, diff_src.data_ptr());
    } else {
      diff_src = at::empty_like(src_dst);
      diff_src_usr_memory = dpcpp_onednn_memory(plain_diff_src_md, engine, diff_src.data_ptr());
    }
  }

  auto strm = GpuStreamManager::Instance().get_stream();
  auto eltwise_bwd = dnnl::eltwise_backward(eltwise_backward_pd);
  if (alg_kind == algorithm::eltwise_logistic_use_dst_for_bwd) {
    DPCPP_ONEDNN_EXEC(eltwise_bwd, strm,
        {{DNNL_ARG_DST, src_dst_usr_memory},
        {DNNL_ARG_DIFF_DST, diff_dst_usr_memory},
        {DNNL_ARG_DIFF_SRC, diff_src_usr_memory}});
  } else {
    DPCPP_ONEDNN_EXEC(eltwise_bwd, strm,
        {{DNNL_ARG_SRC, src_dst_usr_memory},
        {DNNL_ARG_DIFF_DST, diff_dst_usr_memory},
        {DNNL_ARG_DIFF_SRC, diff_src_usr_memory}});
  }
}

} // namespace oneDNN
} // namespace dpcpp
} // namespace at
