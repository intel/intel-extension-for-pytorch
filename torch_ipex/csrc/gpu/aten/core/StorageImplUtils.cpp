#include <c10/util/intrusive_ptr.h>

#include <core/Context.h>
#include <core/DPCPPUtils.h>
#include <core/Exception.h>
#include <core/Memory.h>
#include <utils/General.h>

namespace at {
namespace dpcpp {

void StorageImpl_resize(at::StorageImpl* self, ptrdiff_t size) {
  TORCH_CHECK(size >= 0, "invalid size");
  TORCH_INTERNAL_ASSERT(self->allocator() != nullptr);
  c10::DeviceIndex device;
  AT_DPCPP_CHECK(dpcppGetDevice(&device));

  if (!self->resizable())
    TORCH_CHECK(false, "Trying to resize storage that is not resizable");
  ;

  size_t itemsize = self->itemsize();

  if (size == 0) {
    self->set_data_ptr(
        c10::DataPtr(nullptr, c10::Device(c10::DeviceType::DPCPP, device)));
    self->set_numel(0);
  } else {
    c10::DataPtr data = self->allocator()->allocate(size * itemsize);

    if (self->data_ptr()) {
      dpcppMemcpyAsync(
          data.get(),
          self->data(),
          THMin(self->numel(), size) * itemsize,
          DeviceToDevice);
    }
    self->set_data_ptr(std::move(data));
    self->set_numel(size);
  }
}

int StorageImpl_getDevice(const at::StorageImpl* storage) {
  return storage->device().index();
}

at::StorageImpl* StorageImpl_new(caffe2::TypeMeta data_type) {
  at::StorageImpl* storage =
      c10::make_intrusive<at::StorageImpl>(
          data_type, 0, at::dpcpp::getDPCPPDeviceAllocator(), true)
          .release();
  return storage;
}

} // namespace dpcpp
} // namespace at
