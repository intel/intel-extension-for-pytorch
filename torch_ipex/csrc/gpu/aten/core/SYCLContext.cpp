#include <core/SYCLContext.h>
#include <core/SYCLState.h>
#include <legacy/THSYCLAllocator.h>


namespace at {
namespace sycl {

at::Allocator* getSYCLDeviceAllocator() {
  return THSYCLAllocator_get();
}

} // namespace sycl

} // namespace at
