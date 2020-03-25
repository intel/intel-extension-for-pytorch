#include <torch/csrc/autograd/profiler.h>

#include <utils/Profiler.h>

#include <CL/sycl.hpp>
#include <sstream>


float DPCPPEventStubImpl::elapsed() {
  printf("DPCPPEventStubImpl elapsed ++\n");
  float us;
  event_.wait();
  auto start = event_.template get_profiling_info<
      cl::sycl::info::event_profiling::command_start>();
  auto end = event_.template get_profiling_info<
      cl::sycl::info::event_profiling::command_end>();

  if (end < start) {
    std::stringstream ss;
    ss << __BASE_FILE__ << ":" << __LINE__
       << ": dpcpp profile end time < start time ";
    throw std::runtime_error(ss.str());
  }

  auto duration = end - start;
  // nanoseconds to milliseconds
  us = duration / 1000.0;
  return us;
}

#ifdef DPCPP_PROFILING
void dpcpp_log(std::string name, cl::sycl::event& dpcpp_event) {
  auto stub = std::make_shared<DPCPPEventStubImpl>(dpcpp_event);
  mark_dpcpp(name, stub);
}

struct RegisterDPCPPMethods {
  RegisterDPCPPMethods() {
    static DPCPPProvfilerStubsImpl methods;
    registerDPCPPMethods(&methods);
  }
};

static RegisterDPCPPMethods reg;
#endif
