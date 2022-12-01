#pragma once

#include <ATen/ATen.h>

#include <core/Allocator.h>
#include <core/Memory.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <utils/DPCPP.h>

using namespace xpu::dpcpp;

using mem_desc_t = dnnl::memory::desc;
using mem_dims_t = dnnl::memory::dims;
using mem_dtype_t = dnnl::memory::data_type;
using mem_layout_tag_t = dnnl::memory::format_tag;
using meta_t = mem_desc_t;
using data_t = void*;

namespace at {
namespace AtenIpexTypeXPU {

at::Tensor empty(
    at::IntArrayRef size,
    const at::TensorOptions& options,
    c10::optional<at::MemoryFormat> memory_format);

// PyTorch XPU Tensor context.
// #1. Work as the context instance of at::Storage/at::DataPtr.
// #2. Record oneDNN opaque memory meta.
struct DPCPPTensorContext {
 public:
  using Meta = meta_t;
  using aten_meta_t = struct AtenMeta {
    std::vector<int64_t> sizes_;
    std::vector<int64_t> strides_;
  };

 private:
  data_t data_;
  meta_t meta_;
  aten_meta_t aten_meta_;

 public:
  DPCPPTensorContext() = delete;

  DPCPPTensorContext(const DPCPPTensorContext& ctx)
      : data_(ctx.data()), meta_(ctx.meta()), aten_meta_(ctx.aten_meta()) {}

  // plain tensor
  DPCPPTensorContext(data_t data)
      : data_(data),
        meta_({}, mem_dtype_t::undef, mem_layout_tag_t::undef),
        aten_meta_({{}, {}}) {}

  // block or plain tensor
  DPCPPTensorContext(data_t data, const meta_t& meta)
      : data_(data), meta_(meta), aten_meta_({{}, {}}) {}

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
  bool is_plain() {
    if (meta_.dims().size() == 0 && meta_.data_type() == mem_dtype_t::undef) {
      return true;
    } else if (
        meta_.dims().size() != 0 && meta_.data_type() != mem_dtype_t::undef) {
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
  void set_aten_meta(const aten_meta_t& aten_meta) {
    aten_meta_ = aten_meta;
  }
  data_t data() const {
    return data_;
  }
  meta_t meta() const {
    return meta_;
  }
  aten_meta_t aten_meta() const {
    return aten_meta_;
  }
  mem_dims_t dims() const {
    return meta_.dims();
  }
  mem_dtype_t dtype() const {
    return meta_.data_type();
  }
  mem_dims_t plain_strides() const {
    mem_dims_t dims = meta_.dims();
    mem_dims_t strd(dims.size());
    strd[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--)
      strd[i] = strd[i + 1] * dims[i + 1];
    return strd;
  }

 public:
  static bool is_plain(const at::Tensor& t) {
    if (!t.defined())
      return true;
    auto ctx = get_tensor_ctx(t);
    return ctx.is_plain();
  }

  static void set_aten_meta(
      at::Tensor& t,
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides) {
    auto ctx = (DPCPPTensorContext*)t.unsafeGetTensorImpl()
                   ->storage()
                   .unsafeGetStorageImpl()
                   ->data_ptr()
                   .get_context();
    ctx->set_aten_meta({sizes, strides});
  }

  static aten_meta_t get_aten_meta(at::Tensor& t) {
    auto ctx = (DPCPPTensorContext*)t.unsafeGetTensorImpl()
                   ->storage()
                   .unsafeGetStorageImpl()
                   ->data_ptr()
                   .get_context();
    return ctx->aten_meta();
  }

  static DPCPPTensorContext release_tensor_ctx(const at::Tensor& t) {
    return *(DPCPPTensorContext*)t.unsafeGetTensorImpl()
                ->storage()
                .unsafeGetStorageImpl()
                ->data_ptr()
                .release_context();
  }

  static DPCPPTensorContext get_tensor_ctx(const at::Tensor& t) {
    auto deleter_ptr = t.unsafeGetTensorImpl()
                           ->storage()
                           .unsafeGetStorageImpl()
                           ->data_ptr()
                           .get_deleter();
    if (deleter_ptr == getDeviceAllocator()->raw_deleter()) {
      return *(DPCPPTensorContext*)t.unsafeGetTensorImpl()
                  ->storage()
                  .unsafeGetStorageImpl()
                  ->data_ptr()
                  .get_context();
    } else {
      // If the XPU tensor is not malloced by IPEX, it should be plain format.
      return DPCPPTensorContext{t.unsafeGetTensorImpl()
                                    ->storage()
                                    .unsafeGetStorageImpl()
                                    ->data_ptr()
                                    .get()};
    }
  }

  // set underlying tensor context to given `const` tensor
  static void set_tensor_ctx(const at::Tensor& t, DPCPPTensorContext&& _ctx) {
    data_t cur_raw_data = t.unsafeGetTensorImpl()
                              ->storage()
                              .unsafeGetStorageImpl()
                              ->data_ptr()
                              .get();
    data_t tag_raw_data = _ctx.data();

    auto deleter_ptr = t.unsafeGetTensorImpl()
                           ->storage()
                           .unsafeGetStorageImpl()
                           ->data_ptr()
                           .get_deleter();
    if (deleter_ptr == getDeviceAllocator()->raw_deleter()) {
      // This tensor is malloced by IPEX.
      if (cur_raw_data == tag_raw_data) {
        // It is inplace modification.
        DPCPPTensorContext* cur_ctx =
            static_cast<DPCPPTensorContext*>(t.unsafeGetTensorImpl()
                                                 ->storage()
                                                 .unsafeGetStorageImpl()
                                                 ->data_ptr()
                                                 .get_context());
        *cur_ctx = _ctx;
      } else {
        // It is replacing the memory with a new one.
        auto tag_ctx = new DPCPPTensorContext(_ctx);
        at::DataPtr tag_dptr(
            tag_ctx->data(),
            tag_ctx,
            getDeviceAllocator()->raw_deleter(),
            t.device());
        auto old_dptr = t.unsafeGetTensorImpl()
                            ->storage()
                            .unsafeGetStorageImpl()
                            ->set_data_ptr(std::move(tag_dptr));
      }
    } else {
      // The tensor is not created by IPEX.
      if (cur_raw_data == tag_raw_data) {
        // It is inplace modification.
        TORCH_CHECK(
            false,
            "To set the new format on the shared tensor not allocated by XPU inplacely");
      } else {
        // It is replacing the memory with a new one.
        TORCH_WARN(
            "IPEX relpacing the plain memory format from a shared tensor with block format.",
            "This may break some alias in shared tensor from DLPack");
        auto tag_ctx = new DPCPPTensorContext(_ctx);
        at::DataPtr tag_dptr(
            tag_ctx->data(),
            tag_ctx,
            getDeviceAllocator()->raw_deleter(),
            t.device());
        auto old_dptr = t.unsafeGetTensorImpl()
                            ->storage()
                            .unsafeGetStorageImpl()
                            ->set_data_ptr(std::move(tag_dptr));
      }
    }
  }
};

class DPCPPTensorConvertor {
 public:
  static bool is_opaque_tensor(const at::Tensor& t) {
    auto ctx = DPCPPTensorContext::get_tensor_ctx(t);
    return !ctx.is_plain();
  }

  static at::Tensor to_plain(const at::Tensor& from);
  static at::Tensor to_plain_(at::Tensor& from);
};

} // namespace AtenIpexTypeXPU
} // namespace at
