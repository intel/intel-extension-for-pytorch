#pragma once

#include <list>
#include <memory>

#include <profiler/XPUActivityBuffer.h>
#include <profiler/include/kineto/libkineto.h>

namespace KINETO_NAMESPACE {

struct XPUActivityBuffers {
  std::list<std::unique_ptr<libkineto::CpuTraceBuffer>> cpu;
  std::unique_ptr<XPUActivityBufferMap> gpu;

  template <class T>
  const ITraceActivity& addActivityWrapper(const T& act) {
    wrappers_.push_back(std::make_unique<T>(act));
    return *wrappers_.back().get();
  }

 private:
  std::vector<std::unique_ptr<const ITraceActivity>> wrappers_;
};

} // namespace KINETO_NAMESPACE
