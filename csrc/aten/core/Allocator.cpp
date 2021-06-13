#include <core/CachingAllocator.h>
#include <core/Allocator.h>

namespace xpu {
namespace dpcpp {

at::Allocator* getDeviceAllocator() {
  return dpcpp_getCachingAllocator();
}


} // namespace dpcpp
} // namespace xpu
