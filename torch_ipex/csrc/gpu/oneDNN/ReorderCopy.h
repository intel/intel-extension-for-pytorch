#pragma once

#include <ATen/ATen.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>
#include <tensor/Context.h>
#include "Utils.h"

#include <dnnl.hpp>

#ifdef USE_PRIMITIVE_CACHE
#include <oneDNN/LRUCache.h>
#endif

using namespace dnnl;
using namespace at::AtenIpexTypeXPU;

namespace at {
namespace dpcpp {
namespace oneDNN {

static inline Tensor reordercopy(Tensor& output, const Tensor & src) {
  auto engine = GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  // align to output
  auto output_ctx = DPCPPTensorContext::get_tensor_ctx(output);
  memory::desc output_desc = output_ctx.is_plain() ? memory::desc(
                                                      get_onednn_dims(output),
                                                      get_onednn_dtype(output),
                                                      get_onednn_strides(output))
                                                : output_ctx.meta();
  memory output_mem = dpcpp_onednn_memory(output_desc, engine, output.data_ptr());

  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  memory::desc src_desc = src_ctx.is_plain() ? memory::desc(
                                                    get_onednn_dims(src),
                                                    get_onednn_dtype(src),
                                                    get_onednn_strides(src))
                                              : src_ctx.meta();
  memory src_mem = dpcpp_onednn_memory(src_desc, engine, src.data_ptr());
  if (output_desc != src_desc) {
#ifdef USE_PRIMITIVE_CACHE
    lru_key_t key;
    create_key(key, src_desc, output_desc);
    auto prim = fetch_or_create_m<dnnl::reorder>(key, src_mem, output_mem);
#else
    auto prim = dnnl::reorder(src_mem, output_mem);
#endif
    DPCPP_ONEDNN_EXEC(prim, strm, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, output_mem}});
  }
  return output;
}

}}}

