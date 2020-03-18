#pragma once

#include <c10/core/StorageImpl.h>

namespace at {
namespace dpcpp {

void StorageImpl_resize(at::StorageImpl *self, ptrdiff_t size);
int StorageImpl_getDevice(const at::StorageImpl *storage);
at::StorageImpl *StorageImpl_new(caffe2::TypeMeta data_type);

} // native::
} // at::
