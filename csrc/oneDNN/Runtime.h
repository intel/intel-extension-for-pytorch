#pragma once

#include <ATen/Config.h>

#include <runtime/Context.h>
#include <runtime/DPCPPUtils.h>
#include <core/Device.h>
#include <core/Memory.h>
#include <utils/Profiler.h>
#include <utils/Timer.h>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <vector>


using namespace dnnl;
using namespace xpu::dpcpp;

#define DPCPP_ONEDNN_FORCE_SYNC(stream)             \
  {                                                 \
    static auto force_sync = dpcpp_force_sync();    \
    if (force_sync) {                               \
        (stream).wait();                            \
    }                                               \
  }

#define DPCPP_ONEDNN_EXEC(prim, stream, ...)                                        \
  {                                                                                 \
    static auto verbose = dpcpp_verbose();                                          \
    auto q = dnnl::sycl_interop::get_queue((stream));                               \
    if (verbose) {                                                                  \
      IPEX_TIMER(t, verbose, __func__);                                             \
      DPCPP_BAR_LOG(q, "onednn_kernel",                                             \
          dnnl::sycl_interop::execute((prim), (stream), ##__VA_ARGS__));            \
      t.now("oneDNN execute in sycl_interop");                                      \
      DPCPP_ONEDNN_FORCE_SYNC(stream);                                              \
      t.now("oneDNN stream wait");                                                  \
      q.throw_asynchronous();                                                       \
      t.now("oneDNN throw asynchronous");                                           \
    } else {                                                                        \
      DPCPP_BAR_LOG(q, "onednn_kernel",                                             \
          dnnl::sycl_interop::execute((prim), (stream), ##__VA_ARGS__));            \
      DPCPP_ONEDNN_FORCE_SYNC(stream);                                              \
      q.throw_asynchronous();                                                       \
    }                                                                               \
  }

namespace xpu {
namespace oneDNN {

static inline dnnl::memory dpcpp_onednn_memory(
    dnnl::memory::desc md, dnnl::engine& engine, void* ptr) {
  return dnnl::memory(md, engine, ptr);
}

// GpuEngineManager singleton
struct GpuEngineManager {
  static GpuEngineManager& Instance() {
    static GpuEngineManager myInstance;
    return myInstance;
  }

  engine& get_engine(const Device& device) {
    TORCH_INTERNAL_ASSERT(device.type() == kXPU);
    TORCH_INTERNAL_ASSERT(device.index() < xpu::dpcpp::device_count());
    return *engine_pool[device.index()];
  }

  GpuEngineManager(GpuEngineManager const&) = delete;
  GpuEngineManager& operator=(GpuEngineManager const&) = delete;

 protected:
  GpuEngineManager() {
    int device_count = (int)xpu::dpcpp::device_count();
    TORCH_INTERNAL_ASSERT(device_count > 0);
    for (int i = 0; i < device_count; i++) {
      engine_pool.push_back(std::make_shared<dnnl::engine>(
          dnnl::sycl_interop::make_engine(dpcppGetRawDevice(i), getDeviceContext(i))));
    }
  }
  ~GpuEngineManager() {}

 private:
  std::vector<std::shared_ptr<dnnl::engine>> engine_pool;
};

// GpuStreamManager singleton
struct GpuStreamManager {
  static GpuStreamManager& Instance() {
    static thread_local GpuStreamManager myInstance;
    return myInstance;
  }

#ifdef USE_PERSIST_STREAM
  dnnl::stream& get_stream() {
    int device_index = current_device();
    TORCH_INTERNAL_ASSERT(device_index < xpu::dpcpp::device_count());
    return *stream_pool.at(device_index);
  }
#else
  dnnl::stream get_stream() {
    int device_index = current_device();
    TORCH_INTERNAL_ASSERT(device_index < xpu::dpcpp::device_count());
    return dnnl::sycl_interop::make_stream(
        GpuEngineManager::Instance().get_engine({kXPU, device_index}),
        getDefaultDPCPPStream(device_index).dpcpp_queue());
  }
#endif

  GpuStreamManager(GpuStreamManager const&) = delete;
  GpuStreamManager& operator=(GpuStreamManager const&) = delete;

 protected:
  GpuStreamManager() {
#ifdef USE_PERSIST_STREAM
    int deviceCount = xpu::dpcpp::device_count();
    TORCH_INTERNAL_ASSERT(deviceCount > 0);
    for (DeviceIndex dev = 0; dev < deviceCount; dev++) {
      stream_pool.push_back(std::make_shared<dnnl::stream>(
            dnnl::sycl_interop::make_stream(
              GpuEngineManager::Instance().get_engine({kXPU, dev}),
              getDefaultDPCPPStream(dev).dpcpp_queue())));
    }
#endif
  }
  ~GpuStreamManager() {}

 private:
#ifdef USE_PERSIST_STREAM
  std::vector<std::shared_ptr<dnnl::stream>> stream_pool;
#endif
};

} // namespace dpcpp
} // namespace xpu
