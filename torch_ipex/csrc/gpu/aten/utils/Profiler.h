#pragma once

#include <torch/csrc/autograd/profiler.h>

#include <utils/Env.h>
#include <CL/sycl.hpp>
#include <sstream>
#include <iostream>

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
  virtual float elapsed(XPUEventStub event) override {
    return event->elapsed();
  }
  virtual bool enabled() override {
    return true;
  }
};

static inline void dpcpp_log(std::string name, cl::sycl::event& dpcpp_event) {
  if (dpcpp_profiling() && profilerEnabled()) {
    XPUEventStub dpcpp_evt_stub;
    dpcpp_evt_stub.reset(new DPCPPEventStubImpl(dpcpp_event));
    mark_xpu(std::move(name), dpcpp_evt_stub);
  }
}
