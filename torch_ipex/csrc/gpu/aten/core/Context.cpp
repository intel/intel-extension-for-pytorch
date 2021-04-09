#include <core/Allocator.h>
#include <core/Context.h>
#include <core/DPCPPUtils.h>
#include <core/CachingAllocator.h>

namespace at {
namespace dpcpp {

static std::once_flag initFlag;
static std::vector<std::unique_ptr<DPCPP::context>> gCtxPool;

static void initDeviceContext() {
  int cnt;
  at::dpcpp::dpcppGetDeviceCount(&cnt);

  if (multi_context()) {
    gCtxPool.resize(cnt);
    for (int i = 0; i < cnt; i++) {
      auto dev = at::dpcpp::dpcppGetRawDevice((int64_t)i);
      gCtxPool[i].reset(new DPCPP::context({dev}, at::dpcpp::dpcppAsyncHandler));
    }
  } else {
    DPCPP::vector_class<DPCPP::device> devs;
    for (int i = 0; i < cnt; i++) {
      devs.push_back(at::dpcpp::dpcppGetRawDevice((int64_t)i));
    }

    gCtxPool.resize(1);
    gCtxPool[0].reset(new DPCPP::context(devs, at::dpcpp::dpcppAsyncHandler));
  }
}

void clearDeviceContext() {
  for (auto &ctx: gCtxPool) {
    ctx.reset(NULL);
  }
}

DPCPP::context getDeviceContext(int device_index) {
  // If we use global shared context, we just ignore device_index
  std::call_once(initFlag, initDeviceContext);

  int dev_cnt = -1;
  at::dpcpp::dpcppGetDeviceCount(&dev_cnt);
  AT_ASSERT(device_index < dev_cnt);

  int context_idx;
  if (multi_context()) {
    context_idx = device_index;
  } else {
    context_idx = 0;
  }

  return *gCtxPool[context_idx];
}

at::Allocator* getDPCPPDeviceAllocator() {
  return dpcpp_getCachingAllocator();
}

} // namespace dpcpp
} // namespace at
