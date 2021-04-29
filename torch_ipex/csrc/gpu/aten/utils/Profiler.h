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

static inline cl::sycl::event submit_barrier(cl::sycl::queue& Q) {
  cl::sycl::event e;
  if (dpcpp_profiling() && profilerEnabled()) {
    e = Q.submit_barrier();
  }
  return e;
}

static inline void dpcpp_log(std::string name, cl::sycl::event& start_event, cl::sycl::event& end_event) {
  if (dpcpp_profiling() && profilerEnabled()) {
    XPUEventStub dpcpp_evt_stub;
    dpcpp_evt_stub.reset(new DPCPPEventStubImpl(start_event, end_event));
    mark_xpu(std::move(name), dpcpp_evt_stub);
  }
}

static inline void dpcpp_log(std::string name, cl::sycl::event& dpcpp_event) {
  if (dpcpp_profiling() && profilerEnabled()) {
    XPUEventStub dpcpp_evt_stub;
    dpcpp_evt_stub.reset(new DPCPPEventStubImpl(dpcpp_event));
    mark_xpu(std::move(name), dpcpp_evt_stub);
  }
}
