#include <torch/csrc/autograd/profiler.h>
#include <runtime/Profiler.h>
#include <sstream>


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