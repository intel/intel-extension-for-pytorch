#include <THDP/THSYCLStorage.hpp>
#include <THDP/THSYCLGeneral.h>

#include <THDP/generic/THSYCLStorage.cpp>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLStorage.cpp>
#include <THDP/THSYCLGenerateBoolType.h>

#include <c10/util/intrusive_ptr.h>
#include <c10/dpcpp/SYCLException.h>

void THSYCLStorage_resize(THSYCLState *state, THSYCLStorage *self, ptrdiff_t size) {
  THArgCheck(size >= 0, 2, "invalid size");
  THAssert(self->allocator() != nullptr);
  c10::DeviceIndex device;
  C10_SYCL_CHECK(c10::sycl::syclGetDevice(&device));

  if (!self->resizable())
    THError("Trying to resize storage that is not resizable");;

  size_t itemsize = self->itemsize();

  if (size == 0) {
    self->set_data_ptr(c10::DataPtr(nullptr, c10::Device(c10::DeviceType::SYCL, device)));
    self->set_numel(0);
  } else {
    c10::DataPtr data =
      self->allocator()->allocate(size *itemsize);
    
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


int THSYCLStorage_getDevice(THSYCLState *state, const THSYCLStorage* storage) {
  return storage->device().index();
}

THSYCL_API THSYCLStorage* THSYCLStorage_new(
    THSYCLState* state,
    caffe2::TypeMeta data_type) {
  THSYCLStorage *storage = c10::make_intrusive<c10::StorageImpl> (
      data_type,
      0,
      state->syclDeviceAllocator,
      true).release();
  return storage;
}

