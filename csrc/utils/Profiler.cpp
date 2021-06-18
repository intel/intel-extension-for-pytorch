#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/profiler.h>
#include <c10/core/Allocator.h>
#include <utils/Profiler.h>
#include <utils/Env.h>
#include <sstream>

#if defined(USE_ITT)
#include <itt/itt_wrapper.h>
#endif

using namespace torch::autograd::profiler;

struct DPCPPEventStubImpl : public XPUEventStubBase {
 public:
  DPCPPEventStubImpl() = delete;
  DPCPPEventStubImpl(cl::sycl::event event) : event_(std::move(event)), is_onednn_kernel(false) {};
  DPCPPEventStubImpl(cl::sycl::event start_evt, cl::sycl::event end_evt)
    : event_(std::move(start_evt)), event_end_(std::move(end_evt)), is_onednn_kernel(true) {};
  virtual float elapsed() override;
  virtual ~DPCPPEventStubImpl() = default;

 private:
  cl::sycl::event event_;
  cl::sycl::event event_end_;
  bool is_onednn_kernel; // True for onednn kernel
};

struct DPCPPProfilerStubsImpl : public XPUStubs {
  float elapsed(XPUEventStub event) override {
    return event->elapsed();
  }
  bool enabled() override {
    return true;
  }
  void ittMark(const char* name) override {
#if defined(USE_ITT)
    itt_mark(name);
#else
    AT_ERROR("torch_ipex is not compiled with ITT.");
#endif
  }
  void ittRangePush(const char* name) override {
#if defined(USE_ITT)
    itt_range_push(name);
#else
    AT_ERROR("torch_ipex is not compiled with ITT.");
#endif
  }
  void ittRangePop() override {
#if defined(USE_ITT)
    itt_range_pop();
#else
    AT_ERROR("torch_ipex is not compiled with ITT.");
#endif
  }
};

float DPCPPEventStubImpl::elapsed() {
  float us;
  event_.wait();
  auto start = event_.template get_profiling_info<
      cl::sycl::info::event_profiling::command_start>();
  auto end = event_.template get_profiling_info<
      cl::sycl::info::event_profiling::command_end>();

  if (is_onednn_kernel) {
    event_end_.wait();
    auto start_2 = event_end_.template get_profiling_info<
      cl::sycl::info::event_profiling::command_start>();
    auto end_2 = event_end_.template get_profiling_info<
      cl::sycl::info::event_profiling::command_end>();
    if (start_2 < end) {
      std::stringstream ss;
      ss << __BASE_FILE__ << ":" << __LINE__
         << ": dpcpp onednn profile dummy events overlapped. ";
      throw std::runtime_error(ss.str());
    }
    // nanoseconds to milliseconds
    us = (start_2 - end) / 1000.0;
  } else {
    if (end < start) {
      std::stringstream ss;
      ss << __BASE_FILE__ << ":" << __LINE__
         << ": dpcpp profile end time < start time ";
      throw std::runtime_error(ss.str());
    }
    // nanoseconds to milliseconds
    us = (end - start) / 1000.0;
  }

  return us;
}

struct RegisterDPCPPMethods {
  RegisterDPCPPMethods() {
    static DPCPPProfilerStubsImpl methods;
    registerXPUMethods(&methods);
  }
};

static RegisterDPCPPMethods reg;

bool is_profiler_enabled() {
  return (dpcpp_profiling() && profilerEnabled());
}

void dpcpp_mark(std::string name, cl::sycl::event& event) {
  XPUEventStub dpcpp_evt_stub;
  dpcpp_evt_stub.reset(new DPCPPEventStubImpl(event));
  mark_xpu(std::move(name), dpcpp_evt_stub);
}

void dpcpp_mark(std::string name, cl::sycl::event& start_event, cl::sycl::event& end_event) {
  XPUEventStub dpcpp_evt_stub;
  dpcpp_evt_stub.reset(new DPCPPEventStubImpl(start_event, end_event));
  mark_xpu(std::move(name), dpcpp_evt_stub);
}

void dpcpp_log(std::string name, cl::sycl::event& event) {
  if (is_profiler_enabled()) {
    dpcpp_mark(name, event);
  }
}

void dpcpp_log(std::string name, cl::sycl::event& start_event, cl::sycl::event& end_event) {
  if (is_profiler_enabled()) {
    dpcpp_mark(name, start_event, end_event);
  }
}

void reportMemoryUsage(void* ptr, int64_t alloc_size, at::DeviceIndex device_id) {
  c10::reportMemoryUsageToProfiler(ptr, alloc_size, c10::Device(c10::DeviceType::XPU, device_id));
}
