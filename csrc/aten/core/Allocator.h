#pragma once

#include <c10/core/Allocator.h>
#include <core/CachingAllocator.h>

namespace xpu {
namespace dpcpp {

static inline Allocator* getDeviceAllocator() {
  return dpcpp_getCachingAllocator();
}


} // namespace dpcpp
} // namespace xpu
