#include <core/Context.h>
#include <core/DPCPPUtils.h>
#include <core/Allocator.h>
#include <CL/sycl.hpp>


namespace at {
namespace dpcpp {

static std::once_flag initFlag;
static std::unique_ptr<DP::context> gContext;

static void initGlobalContext() {
  int cnt;
  DP::vector_class<DP::device> devs;
  at::dpcpp::dpcppGetDeviceCount(&cnt);
  for (int i = 0; i < cnt; i++) {
    devs.push_back(at::dpcpp::dpcppGetRawDevice((int64_t)i));
  }

  gContext.reset(new DP::context(devs, at::dpcpp::dpcppAsyncHandler));
}

void clearGlobalContext() {
  gContext.reset(NULL);
}

DP::context getGlobalContext() {
  std::call_once(initFlag, initGlobalContext);
  return *gContext;
}

at::Allocator* getDPCPPDeviceAllocator() {
  return DPCPPAllocator_get();
}

} // namespace dpcpp
} // namespace at
