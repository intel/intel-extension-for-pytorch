#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <profiler/XPUActivity.h>

#include <profiler/LoggerCollector.h>
#include <profiler/include/kineto/GenericTraceActivity.h>
#include <profiler/include/kineto/IActivityProfiler.h>
#include <profiler/include/kineto/ThreadUtil.h>
#include <profiler/include/kineto/TraceSpan.h>
#include <profiler/include/kineto/libkineto.h>
#include <profiler/xpu_output_base.h>

#include <pti/pti_view.h>

namespace KINETO_NAMESPACE {

class Config;
class XPUActivityApi;

struct ConfigDerivedState final {
  ConfigDerivedState() = delete;
  ConfigDerivedState(const Config&);

  bool canStart(
      const std::chrono::time_point<std::chrono::system_clock>& now) const;

  bool isWarmupDone(
      const std::chrono::time_point<std::chrono::system_clock>& now,
      int64_t currentIter) const;

  bool isCollectionDone(
      const std::chrono::time_point<std::chrono::system_clock>& now,
      int64_t currentIter) const;

  const std::set<ActivityType>& profileActivityTypes() const {
    return profileActivityTypes_;
  }

  const std::chrono::time_point<std::chrono::system_clock> profileStartTime()
      const {
    return profileStartTime_;
  }

  const std::chrono::time_point<std::chrono::system_clock> profileEndTime()
      const {
    return profileEndTime_;
  }

  const std::chrono::milliseconds profileDuration() const {
    return profileDuration_;
  }

  int64_t profileStartIteration() const {
    return profileStartIter_;
  }
  int64_t profileEndIteration() const {
    return profileEndIter_;
  }
  bool isProfilingByIteration() const {
    return profilingByIter_;
  }
  bool profileWithPythonStack() const {
    return profileWithStack_;
  }

 private:
  std::set<ActivityType> profileActivityTypes_;
  std::chrono::time_point<std::chrono::system_clock> profileStartTime_;
  std::chrono::time_point<std::chrono::system_clock> profileEndTime_;
  std::chrono::milliseconds profileDuration_;
  std::chrono::seconds profileWarmupDuration_;
  int64_t profileStartIter_{-1};
  int64_t profileEndIter_{-1};
  bool profilingByIter_{false};
  bool profileWithStack_{false};
};

class XPUActivityProfiler {
 public:
  XPUActivityProfiler(XPUActivityApi& onepti, bool cpuOnly);
  XPUActivityProfiler(const XPUActivityProfiler&) = delete;
  XPUActivityProfiler& operator=(const XPUActivityProfiler&) = delete;

  bool isActive() const {
    return currentRunloopState_ != RunloopState::WaitForRequest;
  }

  const std::chrono::time_point<std::chrono::system_clock> performRunLoopStep(
      const std::chrono::time_point<std::chrono::system_clock>& now,
      const std::chrono::time_point<std::chrono::system_clock>& nextWakeupTime,
      int64_t currentIter = -1);

  void setLogger(ActivityLogger* logger) {
    logger_ = logger;
  }

  void startTrace(
      const std::chrono::time_point<std::chrono::system_clock>& now) {
    std::lock_guard<std::mutex> guard(mutex_);
    startTraceInternal(now);
  }

  void stopTrace(
      const std::chrono::time_point<std::chrono::system_clock>& now) {
    std::lock_guard<std::mutex> guard(mutex_);
    stopTraceInternal(now);
  }

  void processTrace(ActivityLogger& logger) {
    std::lock_guard<std::mutex> guard(mutex_);
    processTraceInternal(logger);
  }

  void reset() {
    std::lock_guard<std::mutex> guard(mutex_);
    resetInternal();
  }

  void configure(
      const Config& config,
      const std::chrono::time_point<std::chrono::system_clock>& now);

  void transferCpuTrace(std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace);

  const Config& config() {
    return *config_;
  }

  inline void recordThreadInfo() {
    int32_t sysTid = systemThreadId();
    int32_t tid = threadId();
    int32_t pid = processId();
    std::lock_guard<std::mutex> guard(mutex_);
    recordThreadInfo(sysTid, tid, pid);
  }

  void recordThreadInfo(int32_t sysTid, int32_t tid, int32_t pid) {
    if (resourceInfo_.find({pid, tid}) == resourceInfo_.end()) {
      resourceInfo_.emplace(
          std::make_pair(pid, tid),
          ResourceInfo(
              pid,
              sysTid,
              sysTid, // sortindex
              fmt::format("thread {} ({})", sysTid, getThreadName())));
    }
  }

  void addMetadata(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> guard(mutex_);
    metadata_[key] = value;
  }

  void addChildActivityProfiler(std::unique_ptr<IActivityProfiler> profiler) {
    std::lock_guard<std::mutex> guard(mutex_);
    profilers_.push_back(std::move(profiler));
  }

  // for reducing post processing overhead, filter following runtime
  // ops to be traced down
  static const std::vector<std::string> _traceableRuntimeOps;

  // each runtime record will be checked to trace or not
  static bool isNeededToTrace(const char* name) {
    return std::find(
               _traceableRuntimeOps.begin(),
               _traceableRuntimeOps.end(),
               std::string(name)) != _traceableRuntimeOps.end();
  }

 protected:
  using CpuGpuSpanPair = std::pair<TraceSpan, TraceSpan>;
  static const CpuGpuSpanPair& defaultTraceSpan();

 private:
  class GpuUserEventMap {
   public:
    void insertOrExtendEvent(
        const ITraceActivity& cpuTraceActivity,
        const ITraceActivity& gpuTraceActivity);
    void logEvents(ActivityLogger* logger);

    void clear() {
      streamSpanMap_.clear();
    }

   private:
    using StreamKey = std::pair<int64_t, int64_t>;

    using CorrelationSpanMap =
        std::unordered_map<int64_t, GenericTraceActivity>;
    std::map<StreamKey, CorrelationSpanMap> streamSpanMap_;
  };

  GpuUserEventMap gpuUserEventMap_;
  std::unordered_map<int64_t, const ITraceActivity*> activityMap_;
  std::unordered_map<int64_t, int64_t> cpuCorrelationMap_;
  std::unordered_map<int64_t, const ITraceActivity*> correlatedXPUActivities_;
  std::unordered_map<int64_t, int64_t> userCorrelationMap_;

  struct profilerOverhead {
    int64_t overhead;
    int cntr;
  };

  // void logCudaVersions();

  void startTraceInternal(
      const std::chrono::time_point<std::chrono::system_clock>& now);

  void stopTraceInternal(
      const std::chrono::time_point<std::chrono::system_clock>& now);

  void processTraceInternal(ActivityLogger& logger);

  void resetInternal();

  void finalizeTrace(const Config& config, ActivityLogger& logger);

  void configureChildProfilers();

  void processCpuTrace(
      libkineto::CpuTraceBuffer& cpuTrace,
      ActivityLogger& logger);

  inline void recordStream(int device, int id, const char* postfix) {
    if (resourceInfo_.find({device, id}) == resourceInfo_.end()) {
      resourceInfo_.emplace(
          std::make_pair(device, id),
          ResourceInfo(
              device, id, id, fmt::format("stream {} {}", id, postfix)));
    }
  }

  CpuGpuSpanPair& recordTraceSpan(TraceSpan& span, int gpuOpCount);

  bool iterationTargetMatch(libkineto::CpuTraceBuffer& trace);

  int netId(const std::string& netName);

  const ITraceActivity* linkedActivity(
      int32_t correlationId,
      const std::unordered_map<int64_t, int64_t>& correlationMap);

  const ITraceActivity* cpuActivity(int32_t correlationId);

  void handlePtiActivity(
      const pti_view_record_base* record,
      ActivityLogger* logger);

  void updateGpuNetSpan(const ITraceActivity& gpuOp);
  bool outOfRange(const ITraceActivity& act);
  void handleCorrelationActivity(
      const pti_view_record_external_correlation* correlation);
  void handleRuntimeActivity(
      const pti_view_record_sycl_runtime* activity,
      ActivityLogger* logger);
  void handleOverheadActivity(
      const pti_view_record_overhead* activity,
      ActivityLogger* logger);
  void handleGpuActivity(const ITraceActivity& act, ActivityLogger* logger);
  template <class T>
  void handleGpuActivity(const T* act, ActivityLogger* logger);

  void resetTraceData();

  void addOverheadSample(profilerOverhead& counter, int64_t overhead) {
    counter.overhead += overhead;
    counter.cntr++;
  }
  int64_t getOverhead(const profilerOverhead& counter) {
    if (counter.cntr == 0) {
      return 0;
    }
    return counter.overhead / counter.cntr;
  }

  void checkTimestampOrder(const ITraceActivity* act1);

  std::unique_ptr<const Config> config_;

  std::unique_ptr<ConfigDerivedState> derivedConfig_;

  ActivityLogger* logger_;

  XPUActivityApi& onepti_;

  enum class RunloopState {
    WaitForRequest,
    Warmup,
    CollectTrace,
    ProcessTrace
  };

  std::map<std::string, std::list<CpuGpuSpanPair>> traceSpans_;

  using ActivityTraceMap = std::unordered_map<int64_t, CpuGpuSpanPair*>;
  ActivityTraceMap clientActivityTraceMap_;

  std::map<std::pair<int64_t, int64_t>, ResourceInfo> resourceInfo_;

  std::vector<ActivityLogger::OverheadInfo> overheadInfo_;

  profilerOverhead flushOverhead_;
  profilerOverhead setupOverhead_;

  bool cpuOnly_{false};

  std::mutex mutex_;

  std::atomic<RunloopState> currentRunloopState_{RunloopState::WaitForRequest};

  int64_t captureWindowStartTime_{0};
  int64_t captureWindowEndTime_{0};

  std::map<std::string, int> iterationCountMap_;

  std::unique_ptr<XPUActivityBuffers> traceBuffers_;

  std::unordered_map<std::string, std::string> metadata_;

  std::vector<std::unique_ptr<IActivityProfiler>> profilers_;

  std::vector<std::unique_ptr<IActivityProfilerSession>> sessions_;

  uint32_t resourceOverheadCount_;

  // !USE_GOOGLE_LOG
  std::unique_ptr<LoggerCollector> loggerCollectorMetadata_;
};

} // namespace KINETO_NAMESPACE
