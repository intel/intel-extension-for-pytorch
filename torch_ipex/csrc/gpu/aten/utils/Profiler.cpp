#include <torch/csrc/autograd/profiler.h>

#include <utils/Profiler.h>

#include <CL/sycl.hpp>
#include <sstream>

float DPCPPEventStubImpl::elapsed() {
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

int dpcpp_env(int env_type) {
  static struct {
    int level = [&]() -> int {
      auto env = std::getenv("IPEX_VERBOSE");
      int _level = 0;
      if (env) {
        _level = std::stoi(env, 0, 10);
      }
      std::cout << "IPEX-VERBOSE-LEVEL: " << _level << std::endl;
      return _level;
    } ();

    int force_sync = [&]() -> int {
      auto env = std::getenv("FORCE_SYNC");
      int _force_sync = 0;
      if (env) {
        _force_sync = std::stoi(env, 0, 10);
      }
      std::cout << "Force SYNC: " << _force_sync << std::endl;
      return _force_sync;
    } ();
  } env;

  switch (env_type) {
  case ENV_VERBOSE:
    return env.level;
  case ENV_FORCE_SYNC:
    return env.force_sync;
  default:
    return 0;
  }
}
