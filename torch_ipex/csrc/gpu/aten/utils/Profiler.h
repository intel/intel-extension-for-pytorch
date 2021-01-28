#pragma once

#include <torch/csrc/autograd/profiler.h>

#include <utils/Env.h>
#include <CL/sycl.hpp>
#include <sstream>
#include <iostream>

#if defined(USE_ITT)
#include <itt/itt_wrapper.h>
#endif

using namespace torch::autograd::profiler;

struct DPCPPEventStubImpl : public XPUEventStubBase {
 public:
  DPCPPEventStubImpl() = delete;
  DPCPPEventStubImpl(cl::sycl::event event) : event_(std::move(event)){};
  virtual float elapsed() override;
  virtual ~DPCPPEventStubImpl() = default;

 private:
  cl::sycl::event event_;
};

struct DPCPPProvfilerStubsImpl : public XPUStubs {
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

static inline void dpcpp_log(std::string name, cl::sycl::event& dpcpp_event) {
  if (dpcpp_profiling() && profilerEnabled()) {
    XPUEventStub dpcpp_evt_stub;
    dpcpp_evt_stub.reset(new DPCPPEventStubImpl(dpcpp_event));
    mark_xpu(std::move(name), dpcpp_evt_stub);
  }
}
