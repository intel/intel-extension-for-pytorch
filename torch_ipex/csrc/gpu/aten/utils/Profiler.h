#pragma once

#include <torch/csrc/autograd/profiler.h>

#include <utils/Env.h>
#include <CL/sycl.hpp>
#include <sstream>
#include <chrono>
#include <iostream>

using namespace torch::autograd::profiler;

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

static inline void dpcpp_log(std::string name, cl::sycl::event& dpcpp_event) {
  if (dpcpp_profiling() && profilerEnabled()) {
    auto stub = std::make_shared<DPCPPEventStubImpl>(dpcpp_event);
    mark_dpcpp(std::move(name), std::move(stub));
  }
}
