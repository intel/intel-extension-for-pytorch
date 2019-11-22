#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLStorage.cpp"
#else

#include <c10/util/intrusive_ptr.h>
#include <c10/util/typeid.h>
scalar_t* THSYCLStorage_(data)(THSYCLState *state, const THSYCLStorage *self)
{
  return self->data<scalar_t>();
}

ptrdiff_t THSYCLStorage_(size)(THSYCLState *state, const THSYCLStorage *self)
{
  return THStorage_size(self);
}

int THSYCLStorage_(elementSize)(THSYCLState *state)
{
  return sizeof(scalar_t);
}

void THSYCLStorage_(set)(THSYCLState *state, THSYCLStorage *self, ptrdiff_t index, scalar_t value)
{
  THArgCheck((index >= 0) && (index < self->numel()), 2, "index out of bounds");
  c10::sycl::syclMemcpy(THSYCLStorage_(data)(state, self) + index, &value, sizeof(scalar_t), c10::sycl::HostToDevice);
}

scalar_t THSYCLStorage_(get)(THSYCLState *state, const THSYCLStorage *self, ptrdiff_t index)
{
  THArgCheck((index >= 0) && (index < self->numel()), 2, "index out of bounds");
  scalar_t value;
  c10::sycl::syclMemcpy(&value, THSYCLStorage_(data)(state, self) + index, sizeof (scalar_t), c10::sycl::DeviceToHost);
  return value;
}


THSYCLStorage* THSYCLStorage_(new)(THSYCLState *state)
{
  THStorage* storage = c10::make_intrusive<c10::StorageImpl>(
      caffe2::TypeMeta::Make<scalar_t>(),
      0,
      state->syclDeviceAllocator,
      true).release();
  return storage;
}

THSYCLStorage* THSYCLStorage_(newWithSize)(THSYCLState *state, ptrdiff_t size)
{
  THStorage* storage = c10::make_intrusive<c10::StorageImpl>(
      caffe2::TypeMeta::Make<scalar_t>(),
      size,
      state->syclDeviceAllocator,
      true).release();
  return storage;
}

THSYCLStorage* THSYCLStorage_(newWithAllocator)(THSYCLState *state, ptrdiff_t size,
                                                c10::Allocator* allocator)
{
  THStorage* storage = c10::make_intrusive<c10::StorageImpl>(
      caffe2::TypeMeta::Make<scalar_t>(),
      size,
      allocator,
      true).release();
  return storage;
}

THSYCLStorage* THSYCLStorage_(newWithSize1)(THSYCLState *state, scalar_t data0)
{
  THSYCLStorage *self = THSYCLStorage_(newWithSize)(state, 1);
  THSYCLStorage_(set)(state, self, 0, data0);
  return self;
}

THSYCLStorage* THSYCLStorage_(newWithSize2)(THSYCLState *state, scalar_t data0, scalar_t data1)
{
  THSYCLStorage *self = THSYCLStorage_(newWithSize)(state, 2);
  THSYCLStorage_(set)(state, self, 0, data0);
  THSYCLStorage_(set)(state, self, 1, data1);
  return self;
}

THSYCLStorage* THSYCLStorage_(newWithSize3)(THSYCLState *state, scalar_t data0, scalar_t data1, scalar_t data2)
{
  THSYCLStorage *self = THSYCLStorage_(newWithSize)(state, 3);
  THSYCLStorage_(set)(state, self, 0, data0);
  THSYCLStorage_(set)(state, self, 1, data1);
  THSYCLStorage_(set)(state, self, 2, data2);
  return self;
}

THSYCLStorage* THSYCLStorage_(newWithSize3)(THSYCLState *state, scalar_t data0, scalar_t data1, scalar_t data2, scalar_t data3)
{
  THSYCLStorage *self = THSYCLStorage_(newWithSize)(state, 4);
  THSYCLStorage_(set)(state, self, 0, data0);
  THSYCLStorage_(set)(state, self, 1, data1);
  THSYCLStorage_(set)(state, self, 2, data2);
  THSYCLStorage_(set)(state, self, 3, data3);
  return self;
}



THSYCLStorage* THSYCLStorage_(newWithMapping)(THSYCLState *state, const char *fileName, ptrdiff_t size, int isShared)
{
  THError("not available yet for THSYCLStorage");
  return NULL;
}

THSYCLStorage* THSYCLStorage_(newWithDataAndAllocator)(
    THSYCLState* state,
    c10::DataPtr&& data,
    ptrdiff_t size,
    c10::Allocator* allocator) {
  THSYCLStorage* storage = c10::make_intrusive<c10::StorageImpl>(
      caffe2::TypeMeta::Make<scalar_t>(),
      size,
      std::move(data),
      allocator,
      true).release();
  return storage;
}

void THSYCLStorage_(retain)(THSYCLState *state, THSYCLStorage *self)
{
  THStorage_retain(self);
}

void THSYCLStorage_(free)(THSYCLState *state, THSYCLStorage *self)
{
  THStorage_free(self);
}
#endif


