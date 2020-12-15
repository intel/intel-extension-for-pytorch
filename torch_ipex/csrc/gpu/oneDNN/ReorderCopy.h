#pragma once

#include <ATen/ATen.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>
#include <tensor/Context.h>
#include "Utils.h"

#include <dnnl.hpp>


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
    DPCPP_ONEDNN_EXEC(reorder(src_mem, output_mem), strm,
      {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, output_mem}});
  }
  return output;
}

}}}

