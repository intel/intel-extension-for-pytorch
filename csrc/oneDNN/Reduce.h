#pragma once

#include <ATen/ATen.h>

#include <oneDNN/Runtime.h>
#include <runtime/Utils.h>
#include <tensor/Context.h>

#include "Utils.h"

#include <iostream>

using namespace dnnl;
using namespace at::AtenIpexTypeXPU;

namespace xpu {
namespace oneDNN {

static void reduce(
    const at::Tensor& src,
    at::Tensor& dst,
    const algorithm& aalgorithm,
    float p,
    float eps) {
  TORCH_CHECK(
      src.ndimension() == dst.ndimension(),
      "ndim should be same for src and dst");
  at::Device curDevice = at::Device(at::kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  memory::dims src_dims = get_onednn_dims(src);
  memory::data_type src_dt = get_onednn_dtype(src);

  memory::dims dst_dims = get_onednn_dims(dst);
  memory::data_type dst_dt = get_onednn_dtype(dst);

  auto ndim = src.ndimension();
  memory::format_tag src_format, dst_format;
  src_format = get_dnnl_default_format(ndim);
  dst_format = get_dnnl_default_format(ndim);

  auto desc_src = memory::desc(src_dims, src_dt, src_format);
  auto desc_dst = memory::desc(dst_dims, dst_dt, dst_format);

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, aalgorithm, desc_src, desc_dst, p, eps);
#endif

  auto op_desc = reduction::desc();
  op_desc = reduction::desc(aalgorithm, desc_src, desc_dst, p, eps);

  auto pd = reduction::primitive_desc();
  pd = reduction::primitive_desc(op_desc, engine);

#ifdef USE_PRIMITIVE_CACHE
  auto prim = fetch_or_create_m<reduction>(key, pd);
#else
  auto prim = reduction(pd);
#endif

  const auto src_desc = pd.src_desc();
  const auto dst_desc = pd.dst_desc();

  auto mem_src = dpcpp_onednn_memory(desc_src, engine, src.data_ptr());
  auto mem_dst = dpcpp_onednn_memory(desc_dst, engine, dst.data_ptr());

  DPCPP_ONEDNN_EXEC(
      prim, strm, {{DNNL_ARG_SRC, mem_src}, {DNNL_ARG_DST, mem_dst}});
}
} // namespace oneDNN
} // namespace xpu
