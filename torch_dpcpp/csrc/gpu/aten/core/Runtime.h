#pragma once

#include <ATen/Config.h>

#include <core/SYCLMemory.h>
#include <core/SYCLUtils.h>
#include <core/SYCLContext.h>

#include <vector>
#include <mkldnn.hpp>


using namespace mkldnn;

namespace at { namespace native {

//
// convert vptr to MKL-DNN's SYCL buffer
//
static inline void sycl_set_mkldnn_buffer(void *vptr, mkldnn::memory &memory) {
  //
  //TODO: check size mismatch between vptr and mkldnn::memory
  //

  if (c10::sycl::syclGetBufferMap().get_offset(vptr) == 0) {
    // if offset is 0, which means this sycl_buffer is exact the corresponding one for this vptr, we can safely set it to mkl-dnn sycl API
    auto buffer = c10::sycl::syclGetBufferMap().get_buffer(vptr);
    memory.template set_sycl_buffer<uint8_t, 1>(buffer);
  } else {
    // Currently, memory offset can't have representation in sycl buffer.It's difficult to handle
    // the buffer reclaim of copied buffer and implaced mkldnn operation. If encounter non-zero
    // offset, we currently throw a error. This will not be an issue when usm is available.
    AT_ERROR("the offset of sycl buffer is not 0. We don't support this case.");
  #if 0
    // If offset is not 0, we need to copy a new sycl buffer with correct base addr, then pass to MKL-DNN
    size_t mkldnn_size = memory.get_desc().get_size();
    auto convert_ptr = SYCLmalloc(mkldnn_size, c10::sycl::syclGetBufferMap());
    // TODO: Should we use async copy here?
    // TODO: sycl_buffer life cycle?
    c10::sycl::syclMemcpy(convert_ptr, vptr, mkldnn_size, c10::sycl::DeviceToDevice);

    auto buffer = c10::sycl::syclGetBufferMap().get_buffer(convert_ptr);
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
  } else if (scalar_type == ScalarType::BFloat16){
    return memory::data_type::bf16;
  } else {
    return memory::data_type::f32;
  }
}


// GpuEngineManager singleton
struct GpuEngineManager {
  static GpuEngineManager& Instance() {
    static GpuEngineManager myInstance;
    return myInstance;
  }

  engine& get_engine(const Device &device) {
    AT_ASSERT(device.type() == kSYCL);
    AT_ASSERT(device.index() < c10::sycl::device_count());
    return _gpu_engines[device.index()];
  }

  GpuEngineManager(GpuEngineManager const&) = delete;
  GpuEngineManager& operator=(GpuEngineManager const&) = delete;
protected:
  GpuEngineManager() {
    int device_count = (int) c10::sycl::device_count();
    AT_ASSERT(device_count > 0);
    for (int i = 0; i < device_count; i++) {
      _gpu_engines.push_back({engine::kind::gpu,
                              c10::sycl::getDefaultSYCLStream(i).sycl_queue().get_device(),
                              c10::sycl::getDefaultSYCLStream(i).sycl_queue().get_context()});
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
    int device_count = (int) c10::sycl::device_count();
    AT_ASSERT(device_count > 0 && device_index < device_count);
    return mkldnn::stream(GpuEngineManager::Instance().get_engine({kSYCL, c10::sycl::current_device()}),
                          c10::sycl::getDefaultSYCLStream(device_index).sycl_queue());
  }
  GpuStreamManager(GpuStreamManager const&) = delete;
  GpuStreamManager& operator=(GpuStreamManager const&) = delete;

protected:
  GpuStreamManager() {};
  ~GpuStreamManager() {};
};

}}  // namespace at::native
