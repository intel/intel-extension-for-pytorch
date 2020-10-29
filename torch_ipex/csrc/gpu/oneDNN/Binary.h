#pragma once

#include <ATen/ATen.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>
#include <tensor/Context.h>
#include "Utils.h"

#include <dnnl.hpp>


using namespace dnnl;
using namespace at::AtenIpexTypeDPCPP;

namespace at {
namespace dpcpp {
namespace oneDNN {

template <dnnl::algorithm algo>
static inline Tensor bin(Tensor& output, const Tensor& t1, const Tensor& t2) {
  auto engine = GpuEngineManager::Instance().get_engine({kDPCPP, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto ctx1 = DPCPPTensorContext::get_tensor_ctx(t1);
  auto ctx2 = DPCPPTensorContext::get_tensor_ctx(t2);

  auto tar_ctx = ctx1.is_plain() ? (ctx2.is_plain() ? ctx1 : ctx2) : ctx1;
  auto tar_md = tar_ctx.meta();

  auto md1 = ctx1.is_plain() ?
      memory::desc(get_onednn_dims(t1),
                   get_onednn_dtype(t1),
                   get_onednn_strides(t1)) :
      ctx1.meta();
  auto md2 = ctx2.is_plain() ?
      memory::desc(get_onednn_dims(t2),
                   get_onednn_dtype(t2),
                   get_onednn_strides(t2)) :
      ctx2.meta();

  auto m1_usr = dpcpp_onednn_memory(md1, engine, t1.data_ptr());
  auto m2_usr = dpcpp_onednn_memory(md2, engine, t2.data_ptr());

  Tensor _t1;
  auto m1 = m1_usr;
  if (md1 != tar_md) {
    _t1 = empty_opaque_tensor(tar_md, t1.options(), c10::nullopt);
    m1 = dpcpp_onednn_memory(tar_md, engine, _t1.data_ptr());
    DPCPP_ONEDNN_EXEC(reorder(m1_usr, m1), strm, m1_usr, m1);
  }
  Tensor _t2;
  auto m2 = m2_usr;
  if (md2 != tar_md) {
    _t2 = empty_opaque_tensor(tar_md, t2.options(), c10::nullopt);
    m2 = dpcpp_onednn_memory(tar_md, engine, _t2.data_ptr());
    DPCPP_ONEDNN_EXEC(reorder(m2_usr, m2), strm, m2_usr, m2);
  }

  auto mdo = tar_md;
  if (output.defined()) {
    auto output_ctx = DPCPPTensorContext::get_tensor_ctx(output);
    mdo = output_ctx.is_plain() ?
              memory::desc(get_onednn_dims(output),
                           get_onednn_dtype(output),
                           get_onednn_strides(output)) :
              output_ctx.meta();
  } else {
    output = empty_opaque_tensor(tar_md, t1.options(), c10::nullopt);
  }

  Tensor _output;
  auto mo_usr = dpcpp_onednn_memory(mdo, engine, output.data_ptr());
  auto mo = mo_usr;
  if (mdo != tar_md) {
    _output = at::empty_like(t1);
    mo = dpcpp_onednn_memory(tar_md, engine, _output.data_ptr());
  }

  auto pd = binary::primitive_desc({algo, tar_md, tar_md, tar_md}, engine);
  auto prim = binary(pd);

  prim.execute(strm,
              {{DNNL_ARG_SRC_0, m1},
               {DNNL_ARG_SRC_1, m2},
               {DNNL_ARG_DST, mo}});

  if (mdo != tar_md)
    DPCPP_ONEDNN_EXEC(reorder(mo, mo_usr), strm, mo, mo_usr);

  return output;
}

}}}
