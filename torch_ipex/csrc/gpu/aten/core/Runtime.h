#pragma once

#include <ATen/Config.h>

#include <core/Context.h>
#include <core/DPCPPUtils.h>
#include <core/Memory.h>

#include <mkldnn.hpp>
#include <vector>

using namespace mkldnn;

namespace at {
namespace dpcpp {

//
// convert vptr to MKL-DNN's DPCPP buffer
//
static inline void dpcpp_set_mkldnn_buffer(void *vptr, mkldnn::memory &memory) {
  //
  // TODO: check size mismatch between vptr and mkldnn::memory
  //

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
    AT_ERROR(
        "the offset of dpcpp buffer is not 0. We don't support this case.");
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
  return;
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

// GpuEngineManager singleton
struct GpuEngineManager {
  static GpuEngineManager &Instance() {
    static GpuEngineManager myInstance;
    return myInstance;
  }

  engine &get_engine(const Device &device) {
    AT_ASSERT(device.type() == kDPCPP);
    AT_ASSERT(device.index() < at::dpcpp::device_count());
    return _gpu_engines[device.index()];
  }

  GpuEngineManager(GpuEngineManager const &) = delete;
  GpuEngineManager &operator=(GpuEngineManager const &) = delete;

protected:
  GpuEngineManager() {
    int device_count = (int)at::dpcpp::device_count();
    AT_ASSERT(device_count > 0);
    for (int i = 0; i < device_count; i++) {
      _gpu_engines.push_back({engine::kind::gpu, dpcppGetRawDevice(i),
                              at::dpcpp::getGlobalContext()});
    }
  }
  ~GpuEngineManager() {}

private:
  std::vector<engine> _gpu_engines;
};

// Stream singleton
struct GpuStreamManager {
  static GpuStreamManager &Instance() {
    static thread_local GpuStreamManager myInstance;
    return myInstance;
  };
  stream get_stream(int device_index = 0) {
    int device_count = (int)at::dpcpp::device_count();
    AT_ASSERT(device_count > 0 && device_index < device_count);
    return mkldnn::stream(
        GpuEngineManager::Instance().get_engine({kDPCPP, current_device()}),
        getDefaultDPCPPStream(device_index).dpcpp_queue());
  }
  GpuStreamManager(GpuStreamManager const &) = delete;
  GpuStreamManager &operator=(GpuStreamManager const &) = delete;

protected:
  GpuStreamManager(){};
  ~GpuStreamManager(){};
};
}
} // namespace at::dpcpp
