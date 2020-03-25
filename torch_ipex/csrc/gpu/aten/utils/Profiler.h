#pragma once

#include <torch/csrc/autograd/profiler.h>

#include <CL/sycl.hpp>
#include <sstream>


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

void dpcpp_log(std::string name, cl::sycl::event& dpcpp_event);
