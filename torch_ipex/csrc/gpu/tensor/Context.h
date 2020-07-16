#pragma once

#include <ATen/ATen.h>

#include <core/DPCPPUtils.h>
#include <core/Runtime.h>
#include <core/Allocator.h>
#include <dnnl.hpp>


using namespace at::dpcpp;

using mem_desc_t = dnnl::memory::desc;
using mem_dims_t = dnnl::memory::dims;
using mem_dtype_t = dnnl::memory::data_type;
using mem_layout_tag_t = dnnl::memory::format_tag;
using meta_t = mem_desc_t;
using data_t = void*;

namespace at {
namespace AtenIpexTypeDPCPP {

at::Tensor empty(at::IntArrayRef size,
                 const at::TensorOptions & options,
                 c10::optional<at::MemoryFormat> memory_format);

// creating oneDNN memory object by these ctx data, when reorder is needed.
struct DPCPPTensorContext {
public:
  using Meta = meta_t;

private:
  data_t data_;
  meta_t meta_;

public:
  DPCPPTensorContext() = delete;

  DPCPPTensorContext(const DPCPPTensorContext& ctx) :
      data_(ctx.data()), meta_(ctx.meta()) {}

  // plain tensor
  DPCPPTensorContext(data_t data) :
      data_(data), meta_({}, mem_dtype_t::undef, mem_layout_tag_t::undef) {}

  // block or plain tensor
  DPCPPTensorContext(data_t data, const meta_t& meta) :
      data_(data),
      meta_(meta) {}

  int64_t padded_size() {
    if (is_plain())
      return 0;
    int64_t* padded_dims = meta_.data.padded_dims;
    int64_t ndims = meta_.data.ndims;
    int64_t size = 1;
    for (int64_t dim = 0; dim < ndims; dim++)
      size *= padded_dims[dim];
    return size;
  }

  mem_layout_tag_t get_plain_tag() {
    switch (meta_.dims().size()) {
      case 1:
        return mem_layout_tag_t::a;
      case 2:
        return mem_layout_tag_t::ab;
      case 3:
        return mem_layout_tag_t::abc;
      case 4:
        return mem_layout_tag_t::abcd;
      case 5:
        return mem_layout_tag_t::abcde;
      case 6:
        return mem_layout_tag_t::abcdef;
      default:
        return mem_layout_tag_t::undef;
    }
  }

  bool is_plain() {
    if (meta_.dims().size() == 0 &&
        meta_.data_type() == mem_dtype_t::undef) {
      return true;
    } else if (meta_.dims().size() != 0 &&
               meta_.data_type() != mem_dtype_t::undef) {
      return false;
    } else {
      TORCH_CHECK(1, "fail to check tensor meta ...");
      return false;
    }
  }

  void to_plain() {
    meta_ = meta_t({}, mem_dtype_t::undef, mem_layout_tag_t::undef);
  }

  void set_meta(meta_t meta) {
    meta_ = meta;
  }

  data_t data() const { return data_; }
  meta_t meta() const { return meta_; }
  mem_dims_t dims() const { return meta_.dims(); }
  mem_dtype_t dtype() const { return meta_.data_type(); }

public:
  static bool is_plain(const at::Tensor& t) {
    if (!t.defined())
      return true;
    auto ctx = get_tensor_ctx(t);
    return ctx.is_plain();
  }

  static DPCPPTensorContext release_tensor_ctx(const at::Tensor& t) {
    return *(DPCPPTensorContext*)t.unsafeGetTensorImpl()->
        storage().unsafeGetStorageImpl()->data_ptr().release_context();
  }

  static DPCPPTensorContext get_tensor_ctx(const at::Tensor& t) {
    return *(DPCPPTensorContext*)t.unsafeGetTensorImpl()->
        storage().unsafeGetStorageImpl()->data_ptr().get_context();
  }

  // set underlying tensor context to given `const` tensor
  static void set_tensor_ctx(
      const at::Tensor& t, DPCPPTensorContext&& _ctx) {
    data_t cur_raw_data = t.unsafeGetTensorImpl()->
        storage().unsafeGetStorageImpl()->data_ptr().get();
    data_t tag_raw_data = _ctx.data();

    auto tag_ctx = new DPCPPTensorContext(_ctx);
    at::DataPtr tag_dptr(tag_ctx->data(),
                         tag_ctx,
                         DPCPPAllocator_get()->raw_deleter(),
                         t.device().type());

    // release raw data to avoid auto-free after data_ptr dtor
    DPCPPTensorContext* cur_ctx =
        (DPCPPTensorContext*)t.unsafeGetTensorImpl()->
        storage().unsafeGetStorageImpl()->data_ptr().release_context();

    // swap and old data_ptr dtor
    t.unsafeGetTensorImpl()->
        storage().unsafeGetStorageImpl()->set_data_ptr(std::move(tag_dptr));

    // t->data == ctx->data. raw data is released and reclaim in new data_ptr
    // t->data != ctx->data. old raw data is released and should be deleted by us
    if (cur_raw_data != tag_raw_data)
      DPCPPAllocator_get()->raw_deleter()(cur_ctx);
  }
};

class DPCPPTensorConvertor {
public:
  static bool is_opaque_tensor(const at::Tensor& t) {
    auto ctx = *(static_cast<DPCPPTensorContext*>(
        t.unsafeGetTensorImpl()->storage().data_ptr().get_context()));
    return !ctx.is_plain();
  }

  static bool convert(at::Tensor& to, const at::Tensor& from) {
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
    // FIXME: to decide AB or BA plain format
    mem_desc_t plain_md =
        {opaque_ctx.dims(), opaque_ctx.dtype(), opaque_ctx.get_plain_tag()};
    mem_desc_t from_md = from_is_opaque_tensor ? opaque_md : plain_md;
    mem_desc_t to_md = to_is_opaque_tensor ? opaque_md : plain_md;

    at::Device curDevice = at::Device(at::kDPCPP, current_device());
    auto engine = GpuEngineManager::Instance().get_engine(curDevice);
    auto strm = GpuStreamManager::Instance().get_stream();

    auto from_mem = dpcpp_onednn_memory(from_md, engine, from.data_ptr());
    auto to_mem = dpcpp_onednn_memory(to_md, engine, to.data_ptr());

    DPCPP_ONEDNN_EXEC(dnnl::reorder(from_mem, to_mem), strm, from_mem, to_mem);
    return true;
  }

  static at::Tensor to_plain(const at::Tensor& from) {
    if (!is_opaque_tensor(from))
       return from;

    auto ctx = *(static_cast<DPCPPTensorContext*>(
        from.unsafeGetTensorImpl()->storage().data_ptr().get_context()));
    auto to = at::AtenIpexTypeDPCPP::empty(
        ctx.dims(), from.options(), c10::nullopt);
    convert(to, from);
    return to;
  }

  static at::Tensor to_plain_(at::Tensor& from) {
    if (!is_opaque_tensor(from))
       return from;

    auto to = to_plain(from);

    auto ctx = DPCPPTensorContext::get_tensor_ctx(to);
    DPCPPTensorContext::set_tensor_ctx(from, std::move(ctx));
    return from;
  }
};

} // namespace AtenIpexTypeDPCPP
} // namespace at
