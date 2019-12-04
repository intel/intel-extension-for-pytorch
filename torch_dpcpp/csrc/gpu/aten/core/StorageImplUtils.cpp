#include <c10/util/intrusive_ptr.h>
#include <c10/core/TensorImpl.h>
#include <c10/dpcpp/SYCLUtils.h>
#include <c10/dpcpp/SYCLException.h>
#include <c10/dpcpp/SYCLMemory.h>

#include <core/SYCLContext.h>
#include <utils/General.h>

namespace at { namespace native {

void StorageImpl_resize(at::StorageImpl *self, ptrdiff_t size) {
  TORCH_CHECK(size >= 0, "invalid size");
  TORCH_INTERNAL_ASSERT(self->allocator() != nullptr);
  c10::DeviceIndex device;
  C10_SYCL_CHECK(c10::sycl::syclGetDevice(&device));

  if (!self->resizable())
    TORCH_CHECK(false, "Trying to resize storage that is not resizable");;

  size_t itemsize = self->itemsize();

  if (size == 0) {
    self->set_data_ptr(c10::DataPtr(nullptr, c10::Device(c10::DeviceType::SYCL, device)));
    self->set_numel(0);
  } else {
    c10::DataPtr data = self->allocator()->allocate(size *itemsize);

    if (self->data_ptr()) {
      syclMemcpyAsync(data.get(),
                      self->data(),
                      THMin(self->numel(), size) * itemsize,
                      c10::sycl::DeviceToDevice);
    }
    self->set_data_ptr(std::move(data));
    self->set_numel(size);
  }
}

int StorageImpl_getDevice(const at::StorageImpl* storage) {
  return storage->device().index();
}

at::StorageImpl* StorageImpl_new(caffe2::TypeMeta data_type) {
  at::StorageImpl *storage = c10::make_intrusive<at::StorageImpl> (
      data_type,
      0,
      at::sycl::getSYCLDeviceAllocator(),
      true).release();
  return storage;
}

} // native::
} // at::
