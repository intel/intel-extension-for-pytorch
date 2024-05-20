#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <set>

#include <profiler/XPUActivityBuffer.h>
#include <profiler/include/kineto/ActivityType.h>

#include <pti/pti_view.h>

// struct onepti_Activity;
class UnifiedTracer;

namespace KINETO_NAMESPACE {

using namespace libkineto;

using Onepti_Activity = pti_view_record_base;

class XPUActivityApi {
 public:
  enum CorrelationFlowType { Default, User };

  XPUActivityApi() = default;
  XPUActivityApi(const XPUActivityApi&) = delete;
  XPUActivityApi& operator=(const XPUActivityApi&) = delete;

  virtual ~XPUActivityApi() {}

  static XPUActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enablePtiActivities(const std::set<ActivityType>& selected_activities);
  void disablePtiActivities(const std::set<ActivityType>& selected_activities);
  void clearActivities();

  virtual std::unique_ptr<XPUActivityBufferMap> activityBuffers();

  virtual const std::pair<int, int> processActivities(
      XPUActivityBufferMap&,
      std::function<void(const Onepti_Activity*)> handler);

  void setMaxBufferSize(int size);
  // void setDeviceBufferSize(size_t size);
  // void setDeviceBufferPoolLimit(size_t limit);

  void setDeviceUuidMap(std::vector<std::array<unsigned char, 16>>& uuids);
  int64_t get_device_idx_from_uuid(const uint8_t device_uuid[16]);

  std::atomic_bool stopCollection{false};
  int64_t flushOverhead{0};

 private:
  int maxGpuBufferCount_{0};
  XPUActivityBufferMap allocatedGpuTraceBuffers_;
  std::unique_ptr<XPUActivityBufferMap> readyGpuTraceBuffers_;
  std::mutex mutex_;
  std::atomic<uint32_t> tracingEnabled_{0};
  bool externalCorrelationEnabled_{false};
  std::vector<std::array<unsigned char, 16>> _uuids;

  int processActivitiesForBuffer(
      uint8_t* buf,
      size_t validSize,
      std::function<void(const Onepti_Activity*)> handler);
  static void bufferRequestedTrampoline(uint8_t** buffer, size_t* size);
  static void bufferCompletedTrampoline(
      uint8_t* buffer,
      size_t size,
      size_t validSize);

 protected:
  void bufferRequested(uint8_t** buffer, size_t* size);
  void bufferCompleted(uint8_t* buffer, size_t size, size_t validSize);
};

} // namespace KINETO_NAMESPACE
