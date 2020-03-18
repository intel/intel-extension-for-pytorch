#include <CL/sycl.hpp>
#include <core/Allocator.h>
#include <core/Context.h>
#include <core/DPCPPUtils.h>

namespace at {
namespace dpcpp {

static std::once_flag initFlag;
static std::unique_ptr<DPCPP::context> gContext;

static void initGlobalContext() {
  int cnt;
  DPCPP::vector_class<DPCPP::device> devs;
  at::dpcpp::dpcppGetDeviceCount(&cnt);
  for (int i = 0; i < cnt; i++) {
    devs.push_back(at::dpcpp::dpcppGetRawDevice((int64_t)i));
  }

  gContext.reset(new DPCPP::context(devs, at::dpcpp::dpcppAsyncHandler));
}

void clearGlobalContext() { gContext.reset(NULL); }

DPCPP::context getGlobalContext() {
  std::call_once(initFlag, initGlobalContext);
  return *gContext;
}

at::Allocator *getDPCPPDeviceAllocator() { return DPCPPAllocator_get(); }

} // namespace dpcpp
} // namespace at
