#include <core/Allocator.h>
#include <core/Context.h>
#include <core/DPCPPUtils.h>
#if defined(USE_USM)
#include <core/CachingAllocator.h>
#endif
namespace at {
namespace dpcpp {

static std::once_flag initFlag;
#if !defined(USE_MULTI_CONTEXT)
static std::unique_ptr<DPCPP::context> gContext;
#else
static std::vector<std::unique_ptr<DPCPP::context>> gCtxPool;
#endif

#if !defined(USE_MULTI_CONTEXT)
static void initGlobalContext() {
  int cnt;
  DPCPP::vector_class<DPCPP::device> devs;
  at::dpcpp::dpcppGetDeviceCount(&cnt);
  for (int i = 0; i < cnt; i++) {
    devs.push_back(at::dpcpp::dpcppGetRawDevice((int64_t)i));
  }

  gContext.reset(new DPCPP::context(devs, at::dpcpp::dpcppAsyncHandler));
}
#else
static void initGlobalContext() {
  int cnt;
  at::dpcpp::dpcppGetDeviceCount(&cnt);
  gCtxPool.resize(cnt);
  for (int i = 0; i < cnt; i++) {
    auto dev = at::dpcpp::dpcppGetRawDevice((int64_t)i);
    gCtxPool[i].reset(new DPCPP::context({dev}, at::dpcpp::dpcppAsyncHandler));
  }
}
#endif

#if !defined(USE_MULTI_CONTEXT)
void clearGlobalContext() {
  gContext.reset(NULL);
}
#else
void clearGlobalContext() {
  for (auto &ctx: gCtxPool) {
    ctx.reset(NULL);
  }
}
#endif

#if !defined(USE_MULTI_CONTEXT)
DPCPP::context getGlobalContext() {
  std::call_once(initFlag, initGlobalContext);
  return *gContext;
}
#else
DPCPP::context getGlobalContext() {
  std::call_once(initFlag, initGlobalContext);
  DeviceIndex cur_dev_idx = -1;
  AT_ASSERT(at::dpcpp::dpcppGetDevice(&cur_dev_idx) == DPCPP_SUCCESS);
  return *gCtxPool[cur_dev_idx];
}
#endif

at::Allocator* getDPCPPDeviceAllocator() {
#if defined(USE_USM)
  return dpcpp_getCachingAllocator();
#else
  return DPCPPAllocator_get();
#endif
}

} // namespace dpcpp
} // namespace at
