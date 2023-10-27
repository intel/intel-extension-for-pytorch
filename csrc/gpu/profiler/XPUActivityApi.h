#pragma once

#ifdef USE_KINETO

#include <atomic>
#include <functional>
#include <mutex>
#include <set>

#include "XPUActivityBuffer.h"
#include "kineto/ActivityType.h"

// struct onepti_Activity;
class UnifiedTracer;

namespace KINETO_NAMESPACE {

using namespace libkineto;

class XPUActivityApi {
 public:
  enum CorrelationFlowType { Default, User };

#ifdef USE_ONETRACE
  XPUActivityApi();
#else
  XPUActivityApi() = default;
#endif
  XPUActivityApi(const XPUActivityApi&) = delete;
  XPUActivityApi& operator=(const XPUActivityApi&) = delete;

  virtual ~XPUActivityApi() {}

  static XPUActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableOneptiActivities(
      const std::set<ActivityType>& selected_activities);
  void disableOneptiActivities(
      const std::set<ActivityType>& selected_activities);
  void clearActivities();

  virtual std::unique_ptr<XPUActivityBufferMap> activityBuffers();
  void setDeviceIdMap(const std::vector<std::string>& devices);

#ifdef USE_ONETRACE
  int processActivitiesForBuffer(
      uint8_t* buf,
      size_t validSize,
      std::function<void(const Onepti_Activity*)> handler);

  void startCollecting();
  void stopCollecting();
#endif

  virtual const std::pair<int, int> processActivities(
      XPUActivityBufferMap&,
      std::function<void(const Onepti_Activity*)> handler);

  void setMaxBufferSize(int size);

  std::atomic_bool stopCollection{false};
  int64_t flushOverhead{0};

 private:
  int maxGpuBufferCount_{0};
  XPUActivityBufferMap allocatedGpuTraceBuffers_;
  std::unique_ptr<XPUActivityBufferMap> readyGpuTraceBuffers_;
  std::mutex mutex_;
  std::atomic<uint32_t> tracingEnabled_{0};
  bool externalCorrelationEnabled_{false};
#if defined(USE_ONETRACE)
  static UnifiedTracer* tracer;
#endif
};

} // namespace KINETO_NAMESPACE

#endif
