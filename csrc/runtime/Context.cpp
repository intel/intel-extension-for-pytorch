#include <runtime/Context.h>
#include <runtime/Device.h>
#include <runtime/Exception.h>

namespace xpu {
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
  std::vector<DPCPP::device> devs;
  dpcppGetDeviceCount(&cnt);
  for (int i = 0; i < cnt; i++) {
    devs.push_back(dpcppGetRawDevice(i));
  }

  gContext.reset(new DPCPP::context(devs, dpcppAsyncHandler));
}
#else
static void initDeviceContext() {
  int cnt;
  dpcppGetDeviceCount(&cnt);
  gCtxPool.resize(cnt);
  for (int i = 0; i < cnt; i++) {
    auto dev = dpcppGetRawDevice(i);
    gCtxPool[i].reset(new DPCPP::context({dev}, dpcppAsyncHandler));
  }
}
#endif

#if !defined(USE_MULTI_CONTEXT)
void clearDeviceContext() {
  gContext.reset(NULL);
}
#else
void clearDeviceContext() {
  for (auto& ctx : gCtxPool) {
    ctx.reset(NULL);
  }
}
#endif

#if !defined(USE_MULTI_CONTEXT)
DPCPP::context getDeviceContext(DeviceId device_id) {
  // If we use global shared context, we just ignore device_id
  std::call_once(initFlag, initDeviceContext);
  return *gContext;
}
#else
DPCPP::context getDeviceContext(DeviceId device_id) {
  std::call_once(initFlag, initDeviceContext);
  int dev_cnt = -1;
  dpcppGetDeviceCount(&dev_cnt);
  AT_ASSERT(device_id < dev_cnt);
  return *gCtxPool[device_id];
}
#endif

} // namespace dpcpp
} // namespace xpu
