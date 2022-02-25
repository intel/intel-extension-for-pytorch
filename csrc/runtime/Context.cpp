#include <runtime/Context.h>
#include <runtime/Device.h>
#include <runtime/Exception.h>

namespace xpu {
namespace dpcpp {

#if defined(USE_MULTI_CONTEXT)
static std::once_flag initFlag;
static std::vector<std::unique_ptr<DPCPP::context>> gCtxPool;

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

void clearDeviceContext() {
#if defined(USE_MULTI_CONTEXT)
  for (auto& ctx : gCtxPool) {
    ctx.reset(NULL);
  }
#endif
}

DPCPP::context getDeviceContext(DeviceId device_id) {
#if defined(USE_MULTI_CONTEXT)
  std::call_once(initFlag, initDeviceContext);
  int dev_cnt = -1;
  dpcppGetDeviceCount(&dev_cnt);
  AT_ASSERT(device_id < dev_cnt);
  return *gCtxPool[device_id];
#else
  auto dev = dpcppGetRawDevice(device_id);
  return dev.get_platform().ext_oneapi_get_default_context();
#endif
}

} // namespace dpcpp
} // namespace xpu
