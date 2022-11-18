#pragma once

#include <ATen/ATen.h>
#include <oneDNN/Runtime.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <runtime/Utils.h>
#include <tensor/Context.h>
#include <utils/LRUCache.h>
#include "Utils.h"

using namespace xpu::dpcpp;

namespace xpu {
namespace oneDNN {

static inline void set_params(
    IntArrayRef src_size,
    IntArrayRef dst_size,
    dnnl::memory::dims& src_dims,
    dnnl::memory::dims& dst_dims,
    std::vector<float>& factors,
    int64_t ndims,
    const double& scales_w = 0.0,
    const double& scales_h = 0.0,
    const double& scales_d = 0.0) {
  int64_t n, c, id, ih, iw, od, oh, ow;

  n = src_size[0];
  c = src_size[1];
  id = ih = iw = od = oh = ow = 1;
  if (ndims == 5) {
    od = dst_size[0];
    oh = dst_size[1];
    ow = dst_size[2];

    id = src_size[2];
    ih = src_size[3];
    iw = src_size[4];
  }
  if (ndims == 4) {
    oh = dst_size[0];
    ow = dst_size[1];

    ih = src_size[2];
    iw = src_size[3];
  }
  if (ndims == 3) {
    ow = dst_size[0];
    iw = src_size[2];
  }

  const float depth_scale = scales_d != 0.0
      ? scales_d
      : (std::round((float)od / (float)id * 100) / 100);
  const float height_scale = scales_h != 0.0
      ? scales_h
      : (std::round((float)oh / (float)ih * 100) / 100);
  const float width_scale = scales_w != 0.0
      ? scales_w
      : (std::round((float)ow / (float)iw * 100) / 100);

  src_dims = {n, c};
  dst_dims = {n, c};
  if (ndims == 5) {
    factors.push_back(depth_scale);
    src_dims.push_back(id);
    dst_dims.push_back(od);
  }
  if (ndims >= 4) {
    factors.push_back(height_scale);
    src_dims.push_back(ih);
    dst_dims.push_back(oh);
  }
  if (ndims >= 3) {
    factors.push_back(width_scale);
    src_dims.push_back(iw);
    dst_dims.push_back(ow);
  }
}

static void resample(
    const Tensor& src_,
    Tensor& dst,
    IntArrayRef dst_size,
    algorithm resampling_algorithm,
    const double& scales_w = 0.0,
    const double& scales_h = 0.0,
    const double& scales_d = 0.0) {
  auto strm = GpuStreamManager::Instance().get_stream();
  Device curDevice = Device(kXPU, current_device());
  auto eng = GpuEngineManager::Instance().get_engine(curDevice);

  bool is_customer_scales =
      scales_w != 0.0 || scales_h != 0.0 || scales_d != 0.0;

  int64_t ndims = src_.ndimension();
  IntArrayRef src_size = src_.sizes();
  memory::dims src_dims, dst_dims;
  std::vector<float> factors;
  set_params(
      src_size,
      dst_size,
      src_dims,
      dst_dims,
      factors,
      ndims,
      scales_w,
      scales_h,
      scales_d);

  Tensor src = src_;
  if (is_smf_channels_last(src_)) {
    auto cl_tag = get_cl_tag_by_ndim(ndims);
    if (CHANNELSLAST1D_DPCPP == cl_tag) {
      dst.resize_(dst_dims);
      convert_tensor_to_channels_last_1d(dst);
    } else {
      dst.resize_(dst_dims, get_cl_tag_by_ndim(ndims));
    }
  } else {
    src = src_.contiguous(src_.suggest_memory_format());
    dst.resize_(dst_dims, src_.suggest_memory_format());
  }

  auto data_format = get_dnnl_default_format(ndims, is_smf_channels_last(src_));

  memory::format_tag format_any = memory::format_tag::any;
  memory::data_type data_type = get_onednn_dtype(src);

  std::shared_ptr<memory::desc> dst_md;
  if (!is_customer_scales)
    dst_md.reset(new memory::desc(dst_dims, data_type, format_any));

  auto src_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src);
  auto src_md = src_ctx.is_plain()
      ? memory::desc(src_dims, data_type, data_format)
      : src_ctx.meta();

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  if (!is_customer_scales) {
    create_key(key, resampling_algorithm, factors, src_md, *dst_md);
  } else {
    create_key(key, resampling_algorithm, factors, src_md);
  }
#endif

  auto resampling_desc = resampling_forward::desc(
      prop_kind::forward, resampling_algorithm, factors, src_md, *dst_md);
  auto resampling_pd = resampling_forward::primitive_desc(resampling_desc, eng);

#ifdef USE_PRIMITIVE_CACHE
  auto resample_forward =
      fetch_or_create_m<resampling_forward>(key, resampling_pd);
#else
  auto resample_forward = resampling_forward(resampling_pd);
#endif

  if (!src_ctx.is_plain()) {
    if (src.is_quantized()) {
      auto quantizer = dpcpp_make_per_tensor_affine_quantizer(
          src.q_scale(), src.q_zero_point(), src.scalar_type());
      dst = empty_opaque_qtensor(
          resampling_pd.dst_desc(), c10::nullopt, quantizer);
    } else {
      dst = empty_opaque_tensor(
          resampling_pd.dst_desc(), src.options(), c10::nullopt);
    }
  }
  memory src_memory =
      dpcpp_onednn_memory(resampling_pd.src_desc(), eng, src.data_ptr());
  memory dst_memory =
      dpcpp_onednn_memory(resampling_pd.dst_desc(), eng, dst.data_ptr());

  DPCPP_ONEDNN_EXEC(
      resample_forward,
      strm,
      {{DNNL_ARG_SRC, src_memory}, {DNNL_ARG_DST, dst_memory}});
}

static void resample_backward(
    Tensor& diff_src,
    const Tensor& diff_dst_,
    IntArrayRef src_size,
    IntArrayRef dst_size,
    algorithm resampling_algorithm,
    const double& scales_w = 0.0,
    const double& scales_h = 0.0,
    const double& scales_d = 0.0) {
  auto strm = GpuStreamManager::Instance().get_stream();
  Device curDevice = Device(kXPU, current_device());
  auto eng = GpuEngineManager::Instance().get_engine(curDevice);

  bool is_customer_scales =
      scales_w != 0.0 || scales_h != 0.0 || scales_d != 0.0;

  int64_t ndims = diff_dst_.ndimension();
  memory::dims src_dims, dst_dims;
  std::vector<float> factors;
  set_params(
      src_size,
      dst_size,
      src_dims,
      dst_dims,
      factors,
      ndims,
      scales_w,
      scales_h,
      scales_d);

  Tensor diff_dst;
  if (is_smf_channels_last(diff_dst_)) {
    auto cl_tag = get_cl_tag_by_ndim(ndims);
    if (CHANNELSLAST1D_DPCPP == cl_tag) {
      diff_src.resize_(src_dims);
      convert_tensor_to_channels_last_1d(diff_src);
      auto tmp = diff_dst_.contiguous(at::MemoryFormat::Contiguous);
      diff_dst = convert_tensor_to_channels_last_1d(tmp);
    } else {
      diff_src.resize_(src_dims, get_cl_tag_by_ndim(ndims));
      diff_dst = diff_dst_.contiguous(get_cl_tag_by_ndim(ndims));
    }
  } else {
    diff_src.resize_(src_dims, diff_dst_.suggest_memory_format());
    diff_dst = diff_dst_.contiguous(diff_dst_.suggest_memory_format());
  }

  auto data_format =
      get_dnnl_default_format(ndims, is_smf_channels_last(diff_dst_));

  memory::format_tag format_any = memory::format_tag::any;
  memory::data_type data_type = get_onednn_dtype(diff_dst);

  std::shared_ptr<memory::desc> dst_md;
  auto src_md = memory::desc(src_dims, data_type, data_format);
  auto diff_src_md = memory::desc(src_dims, data_type, format_any);
  if (!is_customer_scales)
    dst_md.reset(new memory::desc(dst_dims, data_type, data_format));

  auto resampling_desc = resampling_forward::desc(
      prop_kind::forward, resampling_algorithm, factors, src_md, *dst_md);
  auto resampling_pd = resampling_forward::primitive_desc(resampling_desc, eng);

  auto diff_dst_ctx =
      at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(diff_dst);
  auto diff_dst_md =
      diff_dst_ctx.is_plain() ? resampling_pd.dst_desc() : diff_dst_ctx.meta();

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  if (!is_customer_scales) {
    create_key(
        key,
        resampling_algorithm,
        factors,
        src_md,
        *dst_md,
        diff_src_md,
        diff_dst_md);
  } else {
    create_key(
        key, resampling_algorithm, factors, src_md, diff_src_md, diff_dst_md);
  }
#endif

  auto resampling_bwd_desc = resampling_backward::desc(
      resampling_algorithm, factors, diff_src_md, diff_dst_md);
  auto resampling_bwd_pd = resampling_backward::primitive_desc(
      resampling_bwd_desc, eng, resampling_pd);
#ifdef USE_PRIMITIVE_CACHE
  auto resampling_bwd =
      fetch_or_create_m<resampling_backward>(key, resampling_bwd_pd);
#else
  auto resampling_bwd = resampling_backward(resampling_bwd_pd);
#endif

  if (!diff_dst_ctx.is_plain()) {
    diff_src = empty_opaque_tensor(
        resampling_bwd_pd.diff_src_desc(), diff_dst.options(), c10::nullopt);
  }
  memory diff_src_memory = dpcpp_onednn_memory(
      resampling_bwd_pd.diff_src_desc(), eng, diff_src.data_ptr());
  memory diff_dst_memory = dpcpp_onednn_memory(
      resampling_bwd_pd.diff_dst_desc(), eng, diff_dst.data_ptr());

  DPCPP_ONEDNN_EXEC(
      resampling_bwd,
      strm,
      {{DNNL_ARG_DIFF_SRC, diff_src_memory},
       {DNNL_ARG_DIFF_DST, diff_dst_memory}});
}

} // namespace oneDNN
} // namespace xpu
