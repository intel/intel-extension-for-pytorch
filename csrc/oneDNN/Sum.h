#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <oneDNN/Runtime.h>
#include <oneDNN/Utils.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>

#include <oneapi/dnnl/dnnl.hpp>

using namespace dnnl;
using namespace at::AtenIpexTypeXPU;

namespace xpu {
namespace oneDNN {

static inline Tensor sum(
    Tensor& output,
    const std::vector<Tensor>& inputs,
    const std::vector<float>& scales) {
  auto engine =
      GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  // align to first inputs
  auto tar = inputs.at(0);
  auto tar_ctx = DPCPPTensorContext::get_tensor_ctx(tar);
  memory::desc tar_desc = tar_ctx.is_plain() ? memory::desc(
                                                   get_onednn_dims(tar),
                                                   get_onednn_dtype(tar),
                                                   get_onednn_strides(tar))
                                             : tar_ctx.meta();

  std::vector<memory::desc> inputs_desc;
  inputs_desc.push_back(tar_desc);

  std::vector<memory> inputs_mem;
  std::vector<Tensor> _curs;
  inputs_mem.push_back(dpcpp_onednn_memory(tar_desc, engine, tar.data_ptr()));
  for (int i = 1; i < inputs.size(); i++) {
    auto cur = inputs.at(i);
    auto cur_ctx = DPCPPTensorContext::get_tensor_ctx(cur);
    memory::desc cur_desc = cur_ctx.is_plain() ? memory::desc(
                                                     get_onednn_dims(cur),
                                                     get_onednn_dtype(cur),
                                                     get_onednn_strides(cur))
                                               : cur_ctx.meta();

    Tensor _cur;
    auto cur_usr_mem = dpcpp_onednn_memory(cur_desc, engine, cur.data_ptr());
    auto cur_mem = cur_usr_mem;
    if (cur_desc != tar_desc) {
      _cur = empty_opaque_tensor(tar_desc, cur.options(), c10::nullopt);
      _curs.push_back(_cur);
      cur_mem = dpcpp_onednn_memory(tar_desc, engine, _cur.data_ptr());
      DPCPP_ONEDNN_EXEC(
          dnnl::reorder(cur_usr_mem, cur_mem),
          strm,
          {{DNNL_ARG_FROM, cur_usr_mem}, {DNNL_ARG_TO, cur_mem}});
    }

    inputs_desc.push_back(tar_desc);
    inputs_mem.push_back(cur_mem);
  }

  auto output_desc = tar_desc;
  if (output.defined()) {
    auto output_ctx = DPCPPTensorContext::get_tensor_ctx(output);
    output_desc = output_ctx.is_plain() ? memory::desc(
                                              get_onednn_dims(output),
                                              get_onednn_dtype(output),
                                              get_onednn_strides(output))
                                        : output_ctx.meta();
  } else {
    output = tar_ctx.is_plain()
        ? at::empty_like(tar)
        : empty_opaque_tensor(tar_desc, tar.options(), c10::nullopt);
  }

  auto output_usr_mem =
      dpcpp_onednn_memory(output_desc, engine, output.data_ptr());
  auto output_mem = output_usr_mem;
  if (output_desc != tar_desc) {
    output_mem = memory(tar_desc, engine);
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(output_usr_mem, output_mem),
        strm,
        {{DNNL_ARG_FROM, output_usr_mem}, {DNNL_ARG_TO, output_mem}});
  }

  auto sum_p = dnnl::sum({output_desc, scales, inputs_desc, engine});

  std::unordered_map<int, memory> args = {{DNNL_ARG_DST, output_mem}};
  for (int i = 0; i < inputs_mem.size(); i++)
    args.insert({DNNL_ARG_MULTIPLE_SRC + i, inputs_mem.at(i)});

  DPCPP_ONEDNN_EXEC(sum_p, strm, args);

  if (output_desc != tar_desc) {
    DPCPP_ONEDNN_EXEC(
        dnnl::reorder(output_mem, output_usr_mem),
        strm,
        {{DNNL_ARG_FROM, output_mem}, {DNNL_ARG_TO, output_usr_mem}});
  }

  return output;
}

} // namespace oneDNN
} // namespace xpu
