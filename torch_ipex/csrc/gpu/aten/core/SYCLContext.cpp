#include <core/SYCLContext.h>
#include <core/SYCLState.h>
#include <core/SYCLUtils.h>
#include <legacy/THSYCLAllocator.h>
#include <CL/sycl.hpp>


namespace at {
namespace sycl {

static std::once_flag initFlag;
static cl::sycl::context gContext;

static void initGlobalContext() {
  int cnt;
  cl::sycl::vector_class<cl::sycl::device> devs;
  c10::sycl::syclGetDeviceCount(&cnt);
  for (int i = 0; i < cnt; i++) {
    devs.push_back(c10::sycl::syclGetRawDevice((int64_t)i));
  }

  gContext = cl::sycl::context(devs, c10::sycl::syclAsyncHandler);
}

cl::sycl::context getGlobalContext() {
  std::call_once(initFlag, initGlobalContext);
  return gContext;
}

at::Allocator* getSYCLDeviceAllocator() {
  return THSYCLAllocator_get();
}

} // namespace sycl

} // namespace at
