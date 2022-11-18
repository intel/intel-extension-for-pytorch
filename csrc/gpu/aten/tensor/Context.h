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

// creating oneDNN memory object by these ctx data, when reorder is needed.
struct DPCPPTensorContext {
 public:
  using Meta = meta_t;

 private:
  data_t data_;
  meta_t meta_;
  std::vector<float> scales_;
  std::vector<int> zero_points_;
  std::vector<long> permution_;

 public:
  DPCPPTensorContext() = delete;

  DPCPPTensorContext(const DPCPPTensorContext& ctx)
      : data_(ctx.data()),
        meta_(ctx.meta()),
        scales_(ctx.scales()),
        zero_points_(ctx.zero_points()),
        permution_(ctx.permution()) {}

  // plain tensor
  DPCPPTensorContext(data_t data)
      : data_(data),
        meta_({}, mem_dtype_t::undef, mem_layout_tag_t::undef),
        scales_({1.f}),
        zero_points_({0}),
        permution_({}) {}

  // block or plain tensor
  DPCPPTensorContext(data_t data, const meta_t& meta)
      : data_(data),
        meta_(meta),
        scales_({1.f}),
        zero_points_({0}),
        permution_({}) {}

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

  void set_scales(std::vector<float> scales) {
    scales_ = scales;
  }

  void set_zero_points(std::vector<int> zero_points) {
    zero_points_ = zero_points;
  }

  data_t data() const {
    return data_;
  }
  meta_t meta() const {
    return meta_;
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
  std::vector<float> scales() const {
    return scales_;
  }
  std::vector<int> zero_points() const {
    return zero_points_;
  }
  void set_permution(const std::vector<long>& permution) {
    permution_ = permution;
  }
  std::vector<long> permution() const {
    return permution_;
  }

 public:
  static bool is_plain(const at::Tensor& t) {
    if (!t.defined())
      return true;
    auto ctx = get_tensor_ctx(t);
    return ctx.is_plain();
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
