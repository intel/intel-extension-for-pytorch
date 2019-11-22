#pragma once

#include <THDP/THSYCLStorage.h>
// Should work with THStorageClass
#include <TH/THStorageFunctions.hpp>
#include <c10/core/ScalarType.h>


#include <c10/dpcpp/SYCLMemory.h>

THSYCL_API THSYCLStorage* THSYCLStorage_new(THSYCLState* state, caffe2::TypeMeta);

THSYCL_API void THSYCLStorage_retain(THSYCLState *state, THSYCLStorage *storage);

THSYCL_API void THSYCLStorage_resize(THSYCLState *state, THSYCLStorage *storage, ptrdiff_t size);

THSYCL_API int THSYCLStorage_getDevice(THSYCLState* state, const THSYCLStorage* storage);

THSYCL_API THSYCLStorage* THSYCLStorage_newWithDataAndAllocator(
    THSYCLState *state, c10::ScalarType scalar_type,
    c10::DataPtr&& data, ptrdiff_t size,
    c10::Allocator* allocator);
