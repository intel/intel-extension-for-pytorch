#pragma once

#include <ATen/Config.h>

#include <core/Context.h>
#include <core/DPCPPUtils.h>
#include <core/Memory.h>
#include <utils/Profiler.h>
#include <utils/Timer.h>

#include <mkldnn.hpp>
#include <vector>

using namespace mkldnn;


#ifdef USE_COMPUTECPP
#define DPCPP_ONEDNN_EXEC(prim, stream, ...)                                   \
  {                                                                            \
    auto verbose = dpcpp_verbose();                                            \
    auto force_sync = dpcpp_force_sync();                                      \
    IPEX_TIMER(t, verbose, __func__);                                          \
    if (force_sync) {                                                          \
      (prim).execute((stream), ##__VA_ARGS__);                                 \
      if (verbose)                                                             \
        t.now("submit");                                                       \
      (stream).wait();                                                         \
      if (verbose)                                                             \
        t.now("wait");                                                         \
    } else {                                                                   \
      (prim).execute((stream), ##__VA_ARGS__);                                 \
      if (verbose)                                                             \
        t.now("submit");                                                       \
    }                                                                          \
  }
#elif defined(USE_DPCPP)
#define DPCPP_ONEDNN_EXEC(prim, stream, ...)                                   \
  {                                                                            \
    DPCPP::event e;                                                            \
    auto verbose = dpcpp_verbose();                                            \
    auto force_sync = dpcpp_force_sync();                                      \
    IPEX_TIMER(t, verbose, __func__);                                          \
    if (force_sync) {                                                          \
      if (verbose) {                                                           \
        e = (prim).execute_sycl((stream), ##__VA_ARGS__);                      \
        t.now("submit");                                                       \
      } else {                                                                 \
        (prim).execute((stream), ##__VA_ARGS__);                               \
      }                                                                        \
      (stream).wait();                                                         \
      if (verbose)                                                             \
        t.now("wait");                                                         \
    } else {                                                                   \
      if (verbose) {                                                           \
        e = (prim).execute_sycl((stream), ##__VA_ARGS__);                      \
        t.now("submit");                                                       \
      } else {                                                                 \
        (prim).execute((stream), ##__VA_ARGS__);                               \
      }                                                                        \
    }                                                                          \
    if (verbose) {                                                             \
      dpcpp_log("dpcpp_kernel", e);                                            \
    }                                                                          \
  }
#else
#error("Unsupported compiler!!!")
#endif

namespace at {
namespace dpcpp {

//
// convert vptr to MKL-DNN's DPCPP buffer
//
static inline void dpcpp_set_mkldnn_buffer(void* vptr, mkldnn::memory& memory) {
  //
  // TODO: check size mismatch between vptr and mkldnn::memory
  //
#if defined(USE_DPCPP)
  auto buffer = make_buffer<uint8_t>(vptr);
  memory.template set_sycl_buffer<uint8_t, 1>(buffer);
#elif defined(USE_COMPUTECPP)
  if (dpcppGetBufferMap().get_offset(vptr) == 0) {
    // if offset is 0, which means this dpcpp_buffer is exact the corresponding
    // one for this vptr, we can safely set it to mkl-dnn dpcpp API
    auto buffer = dpcppGetBufferMap().get_buffer(vptr);
    memory.template set_sycl_buffer<uint8_t, 1>(buffer);
  } else {
    // Currently, memory offset can't have representation in dpcpp buffer.It's
    // difficult to handle
    // the buffer reclaim of copied buffer and implaced mkldnn operation. If
    // encounter non-zero
    // offset, we currently throw a error. This will not be an issue when usm is
    // available.
    TORCH_CHECK(
        0, "the offset of dpcpp buffer is not 0. We don't support this case.");
#if 0
    // If offset is not 0, we need to copy a new dpcpp buffer with correct base addr, then pass to MKL-DNN
    size_t mkldnn_size = memory.get_desc().get_size();
    auto convert_ptr = DPCPPmalloc(mkldnn_size, dpcppGetBufferMap());
    // TODO: Should we use async copy here?
    // TODO: dpcpp_buffer life cycle?
    dpcppMemcpy(convert_ptr, vptr, mkldnn_size, DeviceToDevice);

    auto buffer = dpcppGetBufferMap().get_buffer(convert_ptr);
    memory.template set_sycl_buffer<uint8_t, 1>(buffer);
#endif
  }
#endif
}

//
// convert ScalarType to MKL-DNN's memory type
//
static inline memory::data_type dt_to_dnnl(const ScalarType scalar_type) {
  if (scalar_type == ScalarType::Half) {
    return memory::data_type::f16;
  } else if (scalar_type == ScalarType::BFloat16) {
    return memory::data_type::bf16;
  } else {
    return memory::data_type::f32;
  }
}

static inline memory::format_tag get_dnnl_default_format(int ndims) {
  switch (ndims) {
    case 1:
      return memory::format_tag::a;
    case 2:
      return memory::format_tag::ab;
    case 3:
      return memory::format_tag::abc;
    case 4:
      return memory::format_tag::abcd;
    case 5:
      return memory::format_tag::abcde;
    default:
      return memory::format_tag::any;
  }
}

// GpuEngineManager singleton
struct GpuEngineManager {
  static GpuEngineManager& Instance() {
    static GpuEngineManager myInstance;
    return myInstance;
  }

  engine& get_engine(const Device& device) {
    TORCH_INTERNAL_ASSERT(device.type() == kDPCPP);
    TORCH_INTERNAL_ASSERT(device.index() < at::dpcpp::device_count());
    return _gpu_engines[device.index()];
  }

  GpuEngineManager(GpuEngineManager const&) = delete;
  GpuEngineManager& operator=(GpuEngineManager const&) = delete;

 protected:
  GpuEngineManager() {
    int device_count = (int)at::dpcpp::device_count();
    TORCH_INTERNAL_ASSERT(device_count > 0);
    for (int i = 0; i < device_count; i++) {
      _gpu_engines.push_back({engine::kind::gpu,
                              dpcppGetRawDevice(i),
                              at::dpcpp::getGlobalContext()});
    }
  }
  ~GpuEngineManager() {}

 private:
  std::vector<engine> _gpu_engines;
};

// Stream singleton
struct GpuStreamManager {
  static GpuStreamManager& Instance() {
    static thread_local GpuStreamManager myInstance;
    return myInstance;
  };
  stream get_stream(int device_index = 0) {
    int device_count = (int)at::dpcpp::device_count();
    TORCH_INTERNAL_ASSERT(device_count > 0 && device_index < device_count);
    return mkldnn::stream(
        GpuEngineManager::Instance().get_engine({kDPCPP, current_device()}),
        getDefaultDPCPPStream(device_index).dpcpp_queue());
  }
  GpuStreamManager(GpuStreamManager const&) = delete;
  GpuStreamManager& operator=(GpuStreamManager const&) = delete;

 protected:
  GpuStreamManager(){};
  ~GpuStreamManager(){};
};
} // namespace dpcpp
} // namespace at
