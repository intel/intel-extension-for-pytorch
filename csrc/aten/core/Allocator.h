#pragma once

#include <c10/core/Allocator.h>
#include <runtime/Macros.h>

namespace xpu {
namespace dpcpp {

IPEX_API at::Allocator* getDeviceAllocator();

} // namespace dpcpp
} // namespace xpu
