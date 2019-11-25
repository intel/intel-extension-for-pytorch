#include <core/SYCLContext.h>
#include <THDP/THSYCLGeneral.h>

namespace at {
namespace sycl {

Allocator* getSYCLDeviceAllocator() {
  return at::globalContext().getTHSYCLState()->syclDeviceAllocator;
}

} // namespace sycl

} // namespace at
