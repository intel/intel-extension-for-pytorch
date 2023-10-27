#ifdef USE_KINETO

#include "XPUActivityApi.h"

#include <assert.h>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

#include "Logger.h"
#include "kineto/Config.h"
#include "onepti_activity_api.h"

#if defined(USE_ONETRACE)
#include "trace_options.h"
#include "unified_tracer.h"
#elif defined(USE_PTI)
#endif

using namespace std::chrono;

namespace KINETO_NAMESPACE {

constexpr size_t kBufSize(4 * 1024 * 1024);

#if defined(USE_ONETRACE)
static TraceOptions SetFlags() {
  std::string value;
  uint32_t flags = 0;
  std::string log_file;

  flags |= (1 << TRACE_DEMANGLE);
  flags |= (1 << TRACE_IPEX_CALL_LOGGING);
  flags |= (1 << TRACE_IPEX_DEVICE_STAGES);

  return TraceOptions(flags, log_file);
}

UnifiedTracer* XPUActivityApi::tracer = nullptr;

XPUActivityApi::XPUActivityApi() {
  tracer = UnifiedTracer::Create(SetFlags());
}
#endif

XPUActivityApi& XPUActivityApi::singleton() {
  static XPUActivityApi instance;
  return instance;
}

void XPUActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
#ifdef USE_ONETRACE
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  switch (type) {
    case Default:
      tracer->push_external_correlation_id(
          ONEPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id);
      break;
    case User:
      tracer->push_external_correlation_id(
          ONEPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1, id);
      break;
  }
#endif
}

void XPUActivityApi::popCorrelationID(CorrelationFlowType type) {
#ifdef USE_ONETRACE
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  switch (type) {
    case Default:
      tracer->pop_external_correlation_id(
          ONEPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0);
      break;
    case User:
      tracer->pop_external_correlation_id(
          ONEPTI_EXTERNAL_CORRELATION_KIND_CUSTOM1);
      break;
  }
#endif
}

void XPUActivityApi::enableOneptiActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef USE_ONETRACE
  externalCorrelationEnabled_ = false;
  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::GPU_MEMCPY) {
      // onepti_enable_view(GPU_MEMCPY)
    }
    if (activity == ActivityType::GPU_MEMSET) {
      // onepti_enable_view(GPU_MEMSET)
    }
    if (activity == ActivityType::CONCURRENT_KERNEL) {
      // onepti_enable_view(CONCURRENT_KERNEL)
    }
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
      // onepti_enable_view(EXTERNAL_CORRELATION)
      externalCorrelationEnabled_ = true;
    }
    if (activity == ActivityType::XPU_RUNTIME) {
      // onepti_enable_view(XPU_RUNTIME)
    }
    if (activity == ActivityType::OVERHEAD) {
      // onepti_enable_view(OVERHEAD)
    }
  }

  tracingEnabled_ = 1;
#endif

  stopCollection = false;
}

void XPUActivityApi::disableOneptiActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef USE_ONETRACE
  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::GPU_MEMCPY) {
      // onepti_disable_view(GPU_MEMCPY)
    }
    if (activity == ActivityType::GPU_MEMSET) {
      // onepti_disable_view(GPU_MEMSET)
    }
    if (activity == ActivityType::CONCURRENT_KERNEL) {
      // onepti_disable_view(CONCURRENT_KERNEL)
    }
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
      // onepti_disable_view(EXTERNAL_CORRELATION)
      externalCorrelationEnabled_ = true;
    }
    if (activity == ActivityType::XPU_RUNTIME) {
      // onepti_disable_view(XPU_RUNTIME)
    }
    if (activity == ActivityType::OVERHEAD) {
      // onepti_disable_view(OVERHEAD)
    }
  }
  externalCorrelationEnabled_ = false;
#endif
}

void XPUActivityApi::clearActivities() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedGpuTraceBuffers_.empty()) {
      return;
    }
  }
  std::lock_guard<std::mutex> guard(mutex_);
  readyGpuTraceBuffers_ = nullptr;
}

std::unique_ptr<XPUActivityBufferMap> XPUActivityApi::activityBuffers() {
  // {
  //   std::lock_guard<std::mutex> guard(mutex_);
  //   if (allocatedGpuTraceBuffers_.empty()) {
  //     return nullptr;
  //   }
  // }

#ifdef USE_ONETRACE
  if (!readyGpuTraceBuffers_) {
    readyGpuTraceBuffers_ = std::make_unique<XPUActivityBufferMap>();
    auto buf = std::make_unique<XPUActivityBuffer>(tracer->get_buffer());
    (*readyGpuTraceBuffers_)[buf->data()] = std::move(buf);
  }
  time_point<system_clock> t1 = system_clock::now();
  flushOverhead = duration_cast<microseconds>(system_clock::now() - t1).count();
#endif
  std::lock_guard<std::mutex> guard(mutex_);
  return std::move(readyGpuTraceBuffers_);
}

#ifdef USE_ONETRACE
int XPUActivityApi::processActivitiesForBuffer(
    uint8_t* buf,
    size_t validSize,
    std::function<void(const Onepti_Activity*)> handler) {
  int count = 0;
  if (buf && validSize) {
    Onepti_Activity* record{nullptr};
    while (getNextRecord(buf, validSize, record)) {
      handler(record);
      ++count;
    }
  }
  return count;
}

void XPUActivityApi::startCollecting() {
  tracer->set_active_flag(true);
}

void XPUActivityApi::stopCollecting() {
  tracer->set_active_flag(false);
}
#endif

const std::pair<int, int> XPUActivityApi::processActivities(
    XPUActivityBufferMap& buffers,
    std::function<void(const Onepti_Activity*)> handler) {
  std::pair<int, int> res{0, 0};
#ifdef USE_ONETRACE
  for (auto& pair : buffers) {
    auto& buf = pair.second;
    res.first += processActivitiesForBuffer(buf->data(), buf->size(), handler);
    res.second += buf->size();
  }
#endif
  return res;
}

void XPUActivityApi::setMaxBufferSize(int size) {
  /* actually useless for XPU */
  maxGpuBufferCount_ = 1 + size / kBufSize;
}

void XPUActivityApi::setDeviceIdMap(const std::vector<std::string>& devices) {
  tracer->set_device_id_map(devices);
}

} // namespace KINETO_NAMESPACE

#endif
