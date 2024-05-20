#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <profiler/ActivityLoggerFactory.h>
#include <profiler/InvariantViolations.h>
#include <profiler/LoggerCollector.h>
#include <profiler/XPUActivityApi.h>
#include <profiler/XPUActivityProfiler.h>
#include <profiler/include/kineto/ActivityProfilerInterface.h>
#include <profiler/include/kineto/ActivityTraceInterface.h>

namespace KINETO_NAMESPACE {

class Config;

class XPUActivityProfilerController {
 public:
  explicit XPUActivityProfilerController(bool cpuOnly);
  XPUActivityProfilerController(const XPUActivityProfilerController&) = delete;
  XPUActivityProfilerController& operator=(
      const XPUActivityProfilerController&) = delete;

  ~XPUActivityProfilerController();

  static void setLoggerCollectorFactory(
      std::function<std::unique_ptr<LoggerCollector>()> factory);

  static void addLoggerFactory(
      const std::string& protocol,
      ActivityLoggerFactory::FactoryFunc factory);

  static void setInvariantViolationsLoggerFactory(
      const std::function<std::unique_ptr<InvariantViolationsLogger>()>&
          factory);

  bool canAcceptConfig();
  void acceptConfig(const Config& config);
  void scheduleTrace(const Config& config);

  void prepareTrace(const Config& config);
  void startTrace();
  void step();
  std::unique_ptr<ActivityTraceInterface> stopTrace();

  bool isActive() {
    return profiler_->isActive();
  }

  void transferCpuTrace(std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace) {
    return profiler_->transferCpuTrace(std::move(cpuTrace));
  }

  void recordThreadInfo() {
    profiler_->recordThreadInfo();
  }

  void addChildActivityProfiler(std::unique_ptr<IActivityProfiler> profiler) {
    profiler_->addChildActivityProfiler(std::move(profiler));
  }

  void addMetadata(const std::string& key, const std::string& value);

  void logInvariantViolation(
      const std::string& profile_id,
      const std::string& assertion,
      const std::string& error,
      const std::string& group_profile_id = "");

 private:
  bool shouldActivateIterationConfig(int64_t currentIter);
  bool shouldActivateTimestampConfig(
      const std::chrono::time_point<std::chrono::system_clock>& now);
  void profilerLoop();
  void activateConfig(std::chrono::time_point<std::chrono::system_clock> now);

  std::unique_ptr<Config> asyncRequestConfig_;
  std::mutex asyncConfigLock_;

  std::unique_ptr<XPUActivityProfiler> profiler_;
  std::unique_ptr<ActivityLogger> logger_;
  std::thread* profilerThread_{nullptr};
  std::atomic_bool stopRunloop_{false};
  std::atomic<std::int64_t> iterationCount_{-1};
  // ConfigLoader& configLoader_;
};

} // namespace KINETO_NAMESPACE
