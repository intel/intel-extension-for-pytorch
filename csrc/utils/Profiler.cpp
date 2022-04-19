#include <c10/core/Allocator.h>
#include <include/xpu/Profiler.h>
#include <runtime/Utils.h>
#include <torch/csrc/autograd/profiler_legacy.h>
#include <utils/DPCPP.h>
#include <utils/Helpers.h>
#include <utils/Profiler.h>
#include <utils/Settings.h>
#include <sstream>

using namespace torch::autograd::profiler;

namespace xpu {
namespace dpcpp {

#if defined(USE_PROFILER)
struct DPCPPEventStubImpl : public torch::profiler::impl::KernelEventBase {
 public:
  DPCPPEventStubImpl() = delete;
  DPCPPEventStubImpl(sycl::event event)
      : event_(std::move(event)), is_ext_mark(false){};
  DPCPPEventStubImpl(sycl::event start_evt, sycl::event end_evt)
      : event_(std::move(start_evt)),
        event_end_(std::move(end_evt)),
        is_ext_mark(true){};
  float elapsed();
  float elapsed(DPCPPEventStubImpl& event);
  uint64_t getSubmitTime();
  uint64_t getStartTime();
  uint64_t getEndTime();
  virtual ~DPCPPEventStubImpl() = default;

 private:
  sycl::event event_;
  sycl::event event_end_;
  bool is_ext_mark; // True to mark the external lib kernels
};

struct DPCPPProfilerStubsImpl : public torch::profiler::impl::XPUStubs {
  void record(
      int* device,
      torch::profiler::impl::KernelEventStub* event,
      int64_t* cpu_ns) const override {
    // event has been marked by markKernel, so this record need do nothing
    *device = -1;
  }

  float timeDiff(
      const torch::profiler::impl::KernelEventStub& event,
      const torch::profiler::impl::KernelEventStub& event2) const override {
    DPCPPEventStubImpl* dpcpp_event =
        dynamic_cast<DPCPPEventStubImpl*>(event.get());
    DPCPPEventStubImpl* dpcpp_event2 =
        dynamic_cast<DPCPPEventStubImpl*>(event2.get());
    return dpcpp_event->elapsed(*dpcpp_event2);
  }

  float elapsed(
      const torch::profiler::impl::KernelEventStub& event) const override {
    DPCPPEventStubImpl* dpcpp_event =
        dynamic_cast<DPCPPEventStubImpl*>(event.get());
    return dpcpp_event->elapsed();
  }
  bool enabled() const override {
    return true;
  }
};

float DPCPPEventStubImpl::elapsed() {
  float us;
  event_.wait();
  auto start =
      event_.template get_profiling_info<dpcpp_event_profiling_start>();
  auto end = event_.template get_profiling_info<dpcpp_event_profiling_end>();

  if (is_ext_mark) {
    event_end_.wait();
    auto start_2 =
        event_end_.template get_profiling_info<dpcpp_event_profiling_start>();
    auto end_2 =
        event_end_.template get_profiling_info<dpcpp_event_profiling_end>();
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

uint64_t DPCPPEventStubImpl::getSubmitTime() {
  event_.wait();
  return event_.template get_profiling_info<dpcpp_event_profiling_submit>();
}

uint64_t DPCPPEventStubImpl::getStartTime() {
  event_.wait();
  if (is_ext_mark) {
    return event_.template get_profiling_info<dpcpp_event_profiling_end>();
  }
  return event_.template get_profiling_info<dpcpp_event_profiling_start>();
}

uint64_t DPCPPEventStubImpl::getEndTime() {
  event_.wait();
  if (is_ext_mark) {
    event_end_.wait();
    return event_end_
        .template get_profiling_info<dpcpp_event_profiling_start>();
  }
  return event_.template get_profiling_info<dpcpp_event_profiling_end>();
}

float DPCPPEventStubImpl::elapsed(DPCPPEventStubImpl& other) {
  float us;
  auto start_ns_1 = getStartTime();
  auto start_ns_2 = other.getStartTime();
  us = (start_ns_2 - start_ns_1) / 1000.0;
  return us;
}

struct RegisterDPCPPMethods {
  RegisterDPCPPMethods() {
    static DPCPPProfilerStubsImpl methods;
    registerXPUMethods(&methods);
  }
};

static RegisterDPCPPMethods reg;
#endif

bool is_profiler_enabled() {
#if defined(USE_PROFILER)
  return profilerEnabled();
#else
  return false;
#endif
}

void dpcpp_mark(std::string name, sycl::event& event) {
#if defined(USE_PROFILER)
  torch::profiler::impl::KernelEventStub dpcpp_evt_stub;
  dpcpp_evt_stub.reset(new DPCPPEventStubImpl(event));
  markKernel(std::move(name), dpcpp_evt_stub);
#endif
}

void dpcpp_mark(
    std::string name,
    sycl::event& start_event,
    sycl::event& end_event) {
#if defined(USE_PROFILER)
  torch::profiler::impl::KernelEventStub dpcpp_evt_stub;
  dpcpp_evt_stub.reset(new DPCPPEventStubImpl(start_event, end_event));
  markKernel(std::move(name), dpcpp_evt_stub);
#endif
}

void dpcpp_log(std::string name, sycl::event& event) {
  if (is_profiler_enabled()) {
    dpcpp_mark(name, event);
  }
}

void dpcpp_log(
    std::string name,
    sycl::event& start_event,
    sycl::event& end_event) {
  if (is_profiler_enabled()) {
    dpcpp_mark(name, start_event, end_event);
  }
}

void reportMemoryUsage(
    void* ptr,
    int64_t alloc_size,
    at::DeviceIndex device_id) {
#if defined(USE_PROFILER)
  c10::reportMemoryUsageToProfiler(
      ptr, alloc_size, -1, -1, c10::Device(c10::DeviceType::XPU, device_id));
#endif
}

} // namespace dpcpp

bool is_profiler_enabled() {
  return xpu::dpcpp::is_profiler_enabled();
}

void profiler_record(std::string name, cl::sycl::event& event) {
  return xpu::dpcpp::dpcpp_log(name, event);
}

void profiler_record(
    std::string name,
    cl::sycl::event& start_event,
    cl::sycl::event& end_event) {
  xpu::dpcpp::dpcpp_log(name, start_event, end_event);
}

} // namespace xpu
