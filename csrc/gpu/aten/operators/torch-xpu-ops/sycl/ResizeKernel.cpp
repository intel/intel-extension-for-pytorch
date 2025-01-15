#include "aten/operators/torch-xpu-ops/sycl/ResizeKernel.h"
#include <ATen/Context.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ResizeCommon.h>
#include "aten/operators/torch-xpu-ops/comm/SYCLContext.h"
#include "aten/operators/torch-xpu-ops/comm/XPUGuard.h"
namespace at::native::xpu {

void resize_bytes_xpu(StorageImpl* storage, size_t size_bytes) {
  TORCH_CHECK(
      storage->resizable(), "Trying to resize storage that is not resizable");
  auto allocator = storage->allocator();
  TORCH_CHECK(
      allocator != nullptr, "Trying to resize storage without an allocator");

  c10::Device device = storage->device();

  if (size_bytes == 0) {
    storage->set_data_ptr_noswap(at::DataPtr(nullptr, device));
    storage->set_nbytes(0);
    return;
  }

  c10::xpu::XPUGuard guard(device.index());
  at::DataPtr data = allocator->allocate(size_bytes);
  if (storage->data_ptr()) {
    // at::globalContext().lazyInitDevice(c10::DeviceType::XPU);

    auto q = at::xpu::getCurrentSYCLQueue();
    q.memcpy(
        data.get(), storage->data(), std::min(storage->nbytes(), size_bytes));
  }

  // Destructively overwrite data_ptr
  storage->set_data_ptr_noswap(std::move(data));
  storage->set_nbytes(size_bytes);
}

static inline void maybe_resize_storage_xpu(
    TensorImpl* self,
    size_t new_size_bytes) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in Resize.h)
  if (self->numel() == 0) {
    return;
  }

  const Storage& storage = self->unsafe_storage();
  TORCH_CHECK(storage, "Tensor: invalid null storage");
  if (new_size_bytes > storage.nbytes()) {
    resize_bytes_xpu(storage.unsafeGetStorageImpl(), new_size_bytes);
  }
}

TensorImpl* resize_impl_xpu_(
    TensorImpl* self,
    IntArrayRef size,
    at::OptionalIntArrayRef stride,
    bool device_guard) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  // NB: We don't need to hold the device guard when calling from TH
  at::xpu::OptionalXPUGuard guard;
  if (device_guard) {
    guard.set_index(self->storage().device().index());
  }

  const auto itemsize = self->dtype().itemsize();
  const auto storage_offset = self->storage_offset();
  size_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    storage_size = at::detail::computeStorageNbytes(
        size, *stride, itemsize, storage_offset);
  } else {
    self->set_sizes_contiguous(size);
    storage_size = at::detail::computeStorageNbytesContiguous(
        size, itemsize, storage_offset);
  }
  maybe_resize_storage_xpu(self, storage_size);

  return self;
}

} // namespace at::native::xpu
