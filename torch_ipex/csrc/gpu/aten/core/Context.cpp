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
static void initDeviceContext() {
  int cnt;
  DPCPP::vector_class<DPCPP::device> devs;
  at::dpcpp::dpcppGetDeviceCount(&cnt);
  for (int i = 0; i < cnt; i++) {
    devs.push_back(at::dpcpp::dpcppGetRawDevice((int64_t)i));
  }

  gContext.reset(new DPCPP::context(devs, at::dpcpp::dpcppAsyncHandler));
}
#else
static void initDeviceContext() {
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
void clearDeviceContext() {
  gContext.reset(NULL);
}
#else
void clearDeviceContext() {
  for (auto &ctx: gCtxPool) {
    ctx.reset(NULL);
  }
}
#endif

#if !defined(USE_MULTI_CONTEXT)
DPCPP::context getDeviceContext(int device_index) {
  // If we use global shared context, we just ignore device_index
  std::call_once(initFlag, initDeviceContext);
  return *gContext;
}
#else
DPCPP::context getDeviceContext(int device_index) {
  std::call_once(initFlag, initDeviceContext);
  int dev_cnt = -1;
  at::dpcpp::dpcppGetDeviceCount(&dev_cnt);
  AT_ASSERT(device_index < dev_cnt);
  return *gCtxPool[device_index];
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
