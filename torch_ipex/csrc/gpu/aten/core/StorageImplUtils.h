#pragma once

#include <c10/core/StorageImpl.h>

namespace xpu {
namespace dpcpp {

void StorageImpl_resize(at::StorageImpl* self, ptrdiff_t size_bytes);
int StorageImpl_getDevice(const at::StorageImpl* storage);
at::StorageImpl* StorageImpl_new();

} // namespace dpcpp
} // namespace xpu
