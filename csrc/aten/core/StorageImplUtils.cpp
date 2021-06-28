#include <c10/util/intrusive_ptr.h>

#include <runtime/Utils.h>
#include <runtime/Exception.h>
#include <core/Allocator.h>
#include <core/StorageImplUtils.h>
#include <core/Memory.h>


namespace xpu {
namespace dpcpp {

void StorageImpl_resize(at::StorageImpl* self, ptrdiff_t size_bytes) {
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

int StorageImpl_getDevice(const at::StorageImpl* storage) {
  return storage->device().index();
}

at::StorageImpl* StorageImpl_new() {
  at::StorageImpl* storage =
      c10::make_intrusive<at::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        0,
        getDeviceAllocator(),
        true)
          .release();
  return storage;
}

} // namespace dpcpp
} // namespace at
