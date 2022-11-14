#include <ATen/native/Resize.h>
#include <ATen/quantized/QTensorImpl.h>
#include <c10/core/DeviceType.h>
#include <core/Allocator.h>
#include <core/Device.h>
#include <core/Guard.h>
#include <core/Memory.h>
#include <core/detail/TensorInfo.h>
#include <quantized/Quantizer.h>
#include <runtime/Device.h>
#include <runtime/Exception.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>

namespace at {
namespace AtenIpexTypeXPU {

static void storage_resize(at::StorageImpl* self, ptrdiff_t size_bytes) {
  TORCH_CHECK(size_bytes >= 0, "invalid size");
  TORCH_INTERNAL_ASSERT(self->allocator() != nullptr);
  c10::DeviceIndex device;
  AT_DPCPP_CHECK(dpcppGetDevice(&device));

  if (!self->resizable())
    TORCH_CHECK(false, "Trying to resize storage that is not resizable");

  if (size_bytes == 0) {
    self->set_data_ptr(
        c10::DataPtr(nullptr, c10::Device(c10::DeviceType::XPU, device)));
    self->set_nbytes(0);
  } else {
    c10::DataPtr data = self->allocator()->allocate(size_bytes);

    if (self->data_ptr()) {
      auto nbytes = self->nbytes();
      dpcppMemcpyAsync(
          data.get(),
          self->data(),
          (nbytes < size_bytes ? nbytes : size_bytes),
          DeviceToDevice);
    }
    self->set_data_ptr(std::move(data));
    self->set_nbytes(size_bytes);
  }
}

TensorImpl* resize_impl(
    at::TensorImpl* self,
    at::IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool device_guard) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  OptionalDPCPPGuard guard;
  if (device_guard) {
    guard.set_index(self->storage().device().index());
  }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel.
    storage_size = at::native::storage_size_for(size, *stride);
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }
  // maybe resize storage
  if (storage_size + self->storage_offset() > 0) {
    if (!self->storage().unsafeGetStorageImpl()) {
      TORCH_CHECK(0, "Tensor: invalid null storage");
    }
    int64_t new_size_bytes =
        (storage_size + self->storage_offset()) * self->dtype().itemsize();
    if (new_size_bytes > self->storage().nbytes()) {
      storage_resize(self->storage().unsafeGetStorageImpl(), new_size_bytes);
    }
  }
  return self;
}

bool check_has_opaque_and_no_padding(std::vector<at::Tensor> tlist) {
  std::vector<at::AtenIpexTypeXPU::DPCPPTensorContext> ctx_list;
  for (auto t : tlist)
    ctx_list.push_back(
        at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(t));

  bool all_plain = true;
  for (auto ctx : ctx_list)
    all_plain = all_plain && ctx.is_plain();
  if (all_plain)
    return false;

  for (int i = 0; i < tlist.size(); i++) {
    int64_t padded_numel =
        at::AtenIpexTypeXPU::DPCPPTensorContext(nullptr, ctx_list.at(i).meta())
            .padded_size();
    if (padded_numel != 0 && padded_numel != tlist.at(i).numel())
      return false;
  }

  return true;
}

Tensor share_storage_and_set_strided_as(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  at::Tensor result;
  if (self.is_quantized()) {
    auto quantizer = at::get_qtensorimpl(self)->quantizer();
    result = at::detail::make_tensor<QTensorImpl>(
        Storage(self.storage()), self.key_set(), self.dtype(), quantizer);
  } else {
    result = at::detail::make_tensor<TensorImpl>(
        Storage(self.storage()), self.key_set(), self.dtype());
  }
  at::native::setStrided(result, size, stride, storage_offset);
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
