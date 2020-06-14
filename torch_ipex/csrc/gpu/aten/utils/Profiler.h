#pragma once

#include <torch/csrc/autograd/profiler.h>

#include <CL/sycl.hpp>
#include <sstream>
#include <chrono>
#include <iostream>

using namespace torch::autograd::profiler;

#ifdef DPCPP_PROFILING_KER_PRINT
#define DPCPP_PROF_NOW() std::chrono::steady_clock::now()
#define DPCPP_PROF_KER_PRINT(start, wait, end)                                                      \
    auto __dpcpp_prof_total = std::chrono::duration_cast<std::chrono::microseconds>(end - start);   \
    auto __dpcpp_prof_wait = std::chrono::duration_cast<std::chrono::microseconds>(wait - start);   \
    auto __dpcpp_prof_kernel = std::chrono::duration_cast<std::chrono::microseconds>(end - wait);   \
    std::cout<< "[ " << __FUNCTION__ << " ] in " << __FILE__ << std::endl                           \
      << "Total time: " << __dpcpp_prof_total.count() << " us, "                                    \
      << "Submit time: " << __dpcpp_prof_wait.count() << " us, "                                    \
      << "Kernel time: " << __dpcpp_prof_kernel.count() << " us" << std::endl;
#else
#define DPCPP_PROF_NOW() 0
#define DPCPP_PROF_KER_PRINT(start, wait, end)
#endif

struct DPCPPEventStubImpl : public DPCPPEventStubBase {
 public:
  DPCPPEventStubImpl() = delete;
  DPCPPEventStubImpl(cl::sycl::event event) : event_(event){};
  virtual float elapsed() override;
  virtual ~DPCPPEventStubImpl() = default;

 private:
  cl::sycl::event event_;
};

struct DPCPPProvfilerStubsImpl : public DPCPPStubs {
  virtual float elapsed(DPCPPEventStub event) override {
    return event->elapsed();
  }
  virtual bool enabled() override {
    return true;
  }
};

void dpcpp_log(std::string name, cl::sycl::event& dpcpp_event);
