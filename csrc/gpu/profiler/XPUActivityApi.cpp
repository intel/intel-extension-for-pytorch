#include <profiler/XPUActivityApi.h>

#include <assert.h>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>

#include <profiler/Logger.h>
#include <profiler/include/kineto/Config.h>

using namespace std::chrono;

namespace KINETO_NAMESPACE {

constexpr size_t kBufSize(4 * 1024 * 1024);

XPUActivityApi& XPUActivityApi::singleton() {
  static XPUActivityApi instance;
  return instance;
}

void XPUActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  VLOG(2) << "pushCorrelationID(" << id << ")";
  switch (type) {
    case Default:
      AT_XPU_PTI_CHECK(ptiViewPushExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_0, id));
      break;
    case User:
      AT_XPU_PTI_CHECK(ptiViewPushExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_1, id));
  }
}

void XPUActivityApi::popCorrelationID(CorrelationFlowType type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  switch (type) {
    case Default:
      AT_XPU_PTI_CHECK(ptiViewPopExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_0, nullptr));
      break;
    case User:
      AT_XPU_PTI_CHECK(ptiViewPopExternalCorrelationId(
          pti_view_external_kind::PTI_VIEW_EXTERNAL_KIND_CUSTOM_1, nullptr));
  }
}

static bool nextActivityRecord(
    uint8_t* buffer,
    size_t valid_size,
    Onepti_Activity*& record) {
  pti_result status = ptiViewGetNextRecord(buffer, valid_size, &record);
  if (status != pti_result::PTI_SUCCESS) {
    record = nullptr;
  }
  return record != nullptr;
}

void XPUActivityApi::setMaxBufferSize(int size) {
  maxGpuBufferCount_ = 1 + size / kBufSize;
}

void XPUActivityApi::bufferRequestedTrampoline(uint8_t** buffer, size_t* size) {
  singleton().bufferRequested(buffer, size);
}

void XPUActivityApi::bufferRequested(uint8_t** buffer, size_t* size) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (allocatedGpuTraceBuffers_.size() >= maxGpuBufferCount_) {
    stopCollection = true;
    LOG(WARNING) << "Exceeded max GPU buffer count ("
                 << allocatedGpuTraceBuffers_.size() << " > "
                 << maxGpuBufferCount_ << ") - terminating tracing";
  }

  auto buf = std::make_unique<XPUActivityBuffer>(kBufSize);
  *buffer = buf->data();
  *size = kBufSize;

  allocatedGpuTraceBuffers_[*buffer] = std::move(buf);
}

std::unique_ptr<XPUActivityBufferMap> XPUActivityApi::activityBuffers() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedGpuTraceBuffers_.empty()) {
      return nullptr;
    }
  }

  VLOG(1) << "Flushing GPU activity buffers";
  time_point<system_clock> t1;
  if (VLOG_IS_ON(1)) {
    t1 = system_clock::now();
  }
  AT_XPU_PTI_CHECK(ptiFlushAllViews());
  if (VLOG_IS_ON(1)) {
    flushOverhead =
        duration_cast<microseconds>(system_clock::now() - t1).count();
  }
  std::lock_guard<std::mutex> guard(mutex_);
  return std::move(readyGpuTraceBuffers_);
}

int XPUActivityApi::processActivitiesForBuffer(
    uint8_t* buf,
    size_t validSize,
    std::function<void(const Onepti_Activity*)> handler) {
  int count = 0;
  if (buf && validSize) {
    Onepti_Activity* record{nullptr};
    while (nextActivityRecord(buf, validSize, record)) {
      handler(record);
      ++count;
    }
  }
  return count;
}

const std::pair<int, int> XPUActivityApi::processActivities(
    XPUActivityBufferMap& buffers,
    std::function<void(const Onepti_Activity*)> handler) {
  std::pair<int, int> res{0, 0};
  for (auto& pair : buffers) {
    auto& buf = pair.second;
    res.first += processActivitiesForBuffer(buf->data(), buf->size(), handler);
    res.second += buf->size();
  }
  return res;
}

void XPUActivityApi::clearActivities() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedGpuTraceBuffers_.empty()) {
      return;
    }
  }
  AT_XPU_PTI_CHECK(ptiFlushAllViews());
  std::lock_guard<std::mutex> guard(mutex_);
  readyGpuTraceBuffers_ = nullptr;
}

void XPUActivityApi::bufferCompletedTrampoline(
    uint8_t* buffer,
    size_t size,
    size_t validSize) {
  singleton().bufferCompleted(buffer, size, validSize);
}

void XPUActivityApi::bufferCompleted(
    uint8_t* buffer,
    size_t size,
    size_t validSize) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = allocatedGpuTraceBuffers_.find(buffer);
  if (it == allocatedGpuTraceBuffers_.end()) {
    LOG(ERROR) << "bufferCompleted called with unknown buffer: "
               << (void*)buffer;
    return;
  }

  if (!readyGpuTraceBuffers_) {
    readyGpuTraceBuffers_ = std::make_unique<XPUActivityBufferMap>();
  }
  it->second->setSize(validSize);
  (*readyGpuTraceBuffers_)[it->first] = std::move(it->second);
  allocatedGpuTraceBuffers_.erase(it);
}

void XPUActivityApi::enablePtiActivities(
    const std::set<ActivityType>& selected_activities) {
  AT_XPU_PTI_CHECK(ptiViewSetCallbacks(
      bufferRequestedTrampoline, bufferCompletedTrampoline));

  externalCorrelationEnabled_ = false;
  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::GPU_MEMCPY) {
      AT_XPU_PTI_CHECK(ptiViewEnable(PTI_VIEW_DEVICE_GPU_MEM_COPY));
    }
    if (activity == ActivityType::GPU_MEMSET) {
      AT_XPU_PTI_CHECK(ptiViewEnable(PTI_VIEW_DEVICE_GPU_MEM_FILL));
    }
    if (activity == ActivityType::CONCURRENT_KERNEL) {
      AT_XPU_PTI_CHECK(ptiViewEnable(PTI_VIEW_DEVICE_GPU_KERNEL));
    }
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
      AT_XPU_PTI_CHECK(ptiViewEnable(PTI_VIEW_EXTERNAL_CORRELATION));
      externalCorrelationEnabled_ = true;
    }
    if (activity == ActivityType::XPU_RUNTIME) {
      AT_XPU_PTI_CHECK(ptiViewEnable(PTI_VIEW_SYCL_RUNTIME_CALLS));
    }
    if (activity == ActivityType::OVERHEAD) {
      AT_XPU_PTI_CHECK(ptiViewEnable(PTI_VIEW_COLLECTION_OVERHEAD));
    }
  }

  tracingEnabled_ = 1;

  stopCollection = false;
}

void XPUActivityApi::disablePtiActivities(
    const std::set<ActivityType>& selected_activities) {
  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::GPU_MEMCPY) {
      AT_XPU_PTI_CHECK(ptiViewDisable(PTI_VIEW_DEVICE_GPU_MEM_COPY));
    }
    if (activity == ActivityType::GPU_MEMSET) {
      AT_XPU_PTI_CHECK(ptiViewDisable(PTI_VIEW_DEVICE_GPU_MEM_FILL));
    }
    if (activity == ActivityType::CONCURRENT_KERNEL) {
      AT_XPU_PTI_CHECK(ptiViewDisable(PTI_VIEW_DEVICE_GPU_KERNEL));
    }
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
      AT_XPU_PTI_CHECK(ptiViewDisable(PTI_VIEW_EXTERNAL_CORRELATION));
    }
    if (activity == ActivityType::XPU_RUNTIME) {
      AT_XPU_PTI_CHECK(ptiViewDisable(PTI_VIEW_SYCL_RUNTIME_CALLS));
    }
    if (activity == ActivityType::OVERHEAD) {
      AT_XPU_PTI_CHECK(ptiViewDisable(PTI_VIEW_COLLECTION_OVERHEAD));
    }
  }
  externalCorrelationEnabled_ = false;
}

void XPUActivityApi::setDeviceUuidMap(
    std::vector<std::array<unsigned char, 16>>& uuids) {
  _uuids.assign(uuids.begin(), uuids.end());
}

int64_t XPUActivityApi::get_device_idx_from_uuid(
    const uint8_t device_uuid[16]) {
  std::array<unsigned char, 16> key;
  memcpy(key.data(), device_uuid, 16);
  auto it = std::find(_uuids.begin(), _uuids.end(), key);
  if (it == _uuids.end())
    return static_cast<int64_t>(-1);
  else
    return static_cast<int64_t>(std::distance(_uuids.begin(), it));
}

} // namespace KINETO_NAMESPACE
