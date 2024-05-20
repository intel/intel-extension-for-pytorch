#pragma once

#include <memory>
#include <set>
#include <vector>

#include <profiler/include/kineto/ActivityProfilerInterface.h>
#include <profiler/include/kineto/ActivityType.h>
#include <profiler/include/kineto/ITraceActivity.h>

namespace libkineto {
struct CpuTraceBuffer;
}

namespace KINETO_NAMESPACE {

using namespace libkineto;

class XPUActivityProfilerController;
class Config;

class XPUActivityProfilerProxy : public ActivityProfilerInterface {
 public:
  XPUActivityProfilerProxy(bool cpuOnly);
  ~XPUActivityProfilerProxy() override;

  void init() override;
  bool isInitialized() override {
    return controller_ != nullptr;
  }
  bool isActive() override;

  void scheduleTrace(const std::string& configStr) override;
  void scheduleTrace(const Config& config);

  void prepareTrace(
      const std::set<ActivityType>& activityTypes,
      const std::string& configStr = "") override;
  void startTrace() override;
  std::unique_ptr<ActivityTraceInterface> stopTrace() override;
  void step() override;

  void pushCorrelationId(uint64_t id) override;
  void popCorrelationId() override;
  void transferCpuTrace(std::unique_ptr<CpuTraceBuffer> traceBuffer) override;

  void pushUserCorrelationId(uint64_t) override;
  void popUserCorrelationId() override;

  void recordThreadInfo() override;

  void addMetadata(const std::string& key, const std::string& value) override;

  void addChildActivityProfiler(
      std::unique_ptr<IActivityProfiler> profiler) override;

  void logInvariantViolation(
      const std::string& profile_id,
      const std::string& assertion,
      const std::string& error,
      const std::string& group_profile_id = "") override;

 private:
  bool cpuOnly_{true};
  // ConfigLoader& configLoader_;
  XPUActivityProfilerController* controller_{nullptr};
};

} // namespace KINETO_NAMESPACE
