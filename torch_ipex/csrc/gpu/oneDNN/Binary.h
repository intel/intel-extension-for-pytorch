#pragma once

#include <ATen/ATen.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>
#include <tensor/Context.h>
#include <ATen/AtenIpexTypeXPU.h>
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

template <dnnl::algorithm algo>
static inline Tensor bin(
  Tensor& output,
  const Tensor& t1,
  const Tensor& t2,
  const Tensor t3 = at::Tensor()) {
  auto engine = GpuEngineManager::Instance().get_engine({kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();
  auto ctx1 = DPCPPTensorContext::get_tensor_ctx(t1);
  auto ctx2 = DPCPPTensorContext::get_tensor_ctx(t2);

  auto tar_ctx = ctx1.is_plain() ? (ctx2.is_plain() ? ctx1 : ctx2) : ctx1;
  auto tar_md = (ctx1.is_plain() && ctx2.is_plain()) ?
      memory::desc(get_onednn_dims(t1),
                   get_onednn_dtype(t1),
                   get_onednn_strides(t1)) :
      tar_ctx.meta();

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

  primitive_attr attr;
  post_ops post;
  memory::desc md3;
  memory m3_usr;
  if (t3.defined()) {
    auto ctx3 = DPCPPTensorContext::get_tensor_ctx(t3);
    md3 = ctx3.is_plain() ?
        memory::desc(get_onednn_dims(t3),
                     get_onednn_dtype(t3),
                     get_onednn_strides(t3)) :
        ctx3.meta();
    m3_usr = dpcpp_onednn_memory(md3, engine, t3.data_ptr());
    post.append_binary(dnnl::algorithm::binary_add, md3);
    attr.set_post_ops(post);
  }

  Tensor _t1;
  auto m1 = m1_usr;
  if (md1 != tar_md) {
    _t1 = empty_opaque_tensor(tar_md, t1.options(), c10::nullopt);
    m1 = dpcpp_onednn_memory(tar_md, engine, _t1.data_ptr());
    DPCPP_ONEDNN_EXEC(reorder(m1_usr, m1), strm, {{DNNL_ARG_FROM, m1_usr}, {DNNL_ARG_TO, m1}});
    md1 = tar_md;
  }
  Tensor _t2;
  auto m2 = m2_usr;
  if (md2 != tar_md && t1.sizes() == t2.sizes()) {
    _t2 = empty_opaque_tensor(tar_md, t2.options(), c10::nullopt);
    m2 = dpcpp_onednn_memory(tar_md, engine, _t2.data_ptr());
    DPCPP_ONEDNN_EXEC(reorder(m2_usr, m2), strm, {{DNNL_ARG_FROM, m2_usr}, {DNNL_ARG_TO, m2}});
    md2 = tar_md;
  }

  auto mdo = tar_md;
  if (output.defined()) {
    auto output_ctx = DPCPPTensorContext::get_tensor_ctx(output);
    if(lazy_reorder_enabled()) {
      mdo = output_ctx.is_plain() ? md1 : output_ctx.meta();
    } else {
      mdo = output_ctx.is_plain() ?
                      memory::desc(get_onednn_dims(output),
                                  get_onednn_dtype(output),
                                  get_onednn_strides(output)) :
                      output_ctx.meta();
    }
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

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, tar_md);
#endif
  binary::primitive_desc pd;
  if (t3.defined()) {
    pd = binary::primitive_desc({algo, md1, md2, tar_md}, attr, engine);
  } else {
    pd = binary::primitive_desc({algo, md1, md2, tar_md}, engine);
  }
#ifdef USE_PRIMITIVE_CACHE
  auto prim = fetch_or_create_m<binary>(key, pd);
#else
  auto prim = binary(pd);
#endif

  if (t3.defined()) {
    DPCPP_ONEDNN_EXEC(prim, strm,
                {{DNNL_ARG_SRC_0, m1},
                 {DNNL_ARG_SRC_1, m2},
                 {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, m3_usr},
                 {DNNL_ARG_DST, mo}});
  } else {
    DPCPP_ONEDNN_EXEC(
        prim, strm, {{DNNL_ARG_SRC_0, m1}, {DNNL_ARG_SRC_1, m2}, {DNNL_ARG_DST, mo}});
  }
  if (mdo != tar_md)
    DPCPP_ONEDNN_EXEC(reorder(mo, mo_usr), strm, {{DNNL_ARG_FROM, mo}, {DNNL_ARG_TO, mo_usr}});

  return output;
}

}}}
