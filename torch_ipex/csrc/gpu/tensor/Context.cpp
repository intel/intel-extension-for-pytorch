#include "Context.h"
#include <oneDNN/oneDNN.h>

namespace at {
namespace AtenIpexTypeXPU {

bool DPCPPTensorConvertor::convert(at::Tensor& to, const at::Tensor& from) {
  auto from_ctx = *(static_cast<DPCPPTensorContext*>(
      from.unsafeGetTensorImpl()->storage().data_ptr().get_context()));
  auto to_ctx = *(static_cast<DPCPPTensorContext*>(
      to.unsafeGetTensorImpl()->storage().data_ptr().get_context()));

  auto to_is_opaque_tensor = is_opaque_tensor(to);
  auto from_is_opaque_tensor = is_opaque_tensor(from);

  if (!to_is_opaque_tensor && !from_is_opaque_tensor)
    return false;

  auto opaque_ctx = from_is_opaque_tensor ? from_ctx : to_ctx;
  mem_desc_t opaque_md = {opaque_ctx.meta().data};
  mem_desc_t plain_md = {opaque_ctx.dims(),
                         opaque_ctx.dtype(),
                         opaque_ctx.plain_strides()};
  mem_desc_t from_md = from_is_opaque_tensor ? opaque_md : plain_md;
  mem_desc_t to_md = to_is_opaque_tensor ? opaque_md : plain_md;

  at::Device curDevice = at::Device(at::kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  auto from_mem = dpcpp_onednn_memory(from_md, engine, from.data_ptr());
  auto to_mem = dpcpp_onednn_memory(to_md, engine, to.data_ptr());

#ifdef USE_PRIMITIVE_CACHE
  lru_key_t key;
  create_key(key, from_md, to_md);
  auto reorder_p = fetch_or_create_m<dnnl::reorder>(key, from_mem, to_mem);
#else
  auto reorder_p = dnnl::reorder(from_mem, to_mem);
#endif
  DPCPP_ONEDNN_EXEC(reorder_p, strm, {{DNNL_ARG_FROM, from_mem}, {DNNL_ARG_TO, to_mem}});

  return true;
}

at::Tensor DPCPPTensorConvertor::to_plain(const at::Tensor& from) {
  if (!is_opaque_tensor(from))
    return from;

  auto ctx = *(static_cast<DPCPPTensorContext*>(
      from.unsafeGetTensorImpl()->storage().data_ptr().get_context()));

  // case-1. int32 opaque tensor in plain fmt
  if (from.scalar_type() == at::ScalarType::Long) {
    mem_desc_t opaque_md = {ctx.meta().data};
    // FIXME: to decide AB or BA plain format
    mem_desc_t plain_md = {ctx.dims(), ctx.dtype(), ctx.get_plain_tag()};
    if (opaque_md == plain_md) {
      Tensor to = at::empty(ctx.dims(), from.options(), c10::nullopt);
      dpcppMemoryCopyType(to.data_ptr<int64_t>(), (int32_t*)from.data_ptr(), from.numel());
      return to;
    }
  }

  auto options = from.options();
  if (from.scalar_type() == at::ScalarType::Long)
    options = options.dtype(kInt);

  // reorder to plain based on current shape
  auto to = at::AtenIpexTypeXPU::empty(ctx.dims(), options, c10::nullopt);
  convert(to, from);

  // permute shape to original shape
  if (!ctx.permution().empty())
    to = at::native::permute(to, IntArrayRef(ctx.permution())).contiguous();

  // case-2. int32 opaque tensor in block fmt
  // 1. convert to plain 2. copy to int64
  if (from.scalar_type() == at::ScalarType::Long) {
    Tensor to_ = at::empty(ctx.dims(), from.options(), c10::nullopt);
    dpcppMemoryCopyType(to_.data_ptr<int64_t>(), to.data_ptr<int32_t>(), to.numel());
    to = to_;
  }

  return to;
}

at::Tensor DPCPPTensorConvertor::to_plain_(at::Tensor& from) {
  if (!is_opaque_tensor(from))
    return from;

  auto to = to_plain(from);

  auto ctx = DPCPPTensorContext::get_tensor_ctx(to);
  DPCPPTensorContext::set_tensor_ctx(from, std::move(ctx));
  return from;
}

}}
