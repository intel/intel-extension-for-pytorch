#pragma once

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include <profiler/include/kineto/Config.h>
#include <profiler/include/kineto/GenericTraceActivity.h>
#include <profiler/xpu_output_base.h>

namespace KINETO_NAMESPACE {

class Config;

class XPUMemoryTraceLogger : public ActivityLogger {
 public:
  XPUMemoryTraceLogger(const Config& config) : config_(config.clone()) {
    activities_.reserve(100000);
  }

  void handleDeviceInfo(const DeviceInfo& info, uint64_t time) override {
    deviceInfoList_.emplace_back(info, time);
  }

  void handleResourceInfo(const ResourceInfo& info, int64_t time) override {
    resourceInfoList_.emplace_back(info, time);
  }

  void handleOverheadInfo(const OverheadInfo& info, int64_t time) override {}

  void handleTraceSpan(const TraceSpan& span) override {
    // Handled separately
  }

  template <class T>
  void addActivityWrapper(const T& act) {
    wrappers_.push_back(std::make_unique<T>(act));
    activities_.push_back(wrappers_.back().get());
  }

  void handleActivity(const ITraceActivity& activity) override {
    activities_.push_back(&activity);
  }
  void handleGenericActivity(const GenericTraceActivity& activity) override {
    addActivityWrapper(activity);
  }

  void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata) override {
    metadata_ = metadata;
  }

  void finalizeTrace(
      const Config& config,
      std::unique_ptr<XPUActivityBuffers> buffers,
      int64_t endTime,
      std::unordered_map<std::string, std::vector<std::string>>& metadata)
      override {
    buffers_ = std::move(buffers);
    endTime_ = endTime;
  }

  const std::vector<const ITraceActivity*>* traceActivities() {
    return &activities_;
  }

  void log(ActivityLogger& logger) {
    logger.handleTraceStart(metadata_);
    for (auto& activity : activities_) {
      activity->log(logger);
    }
    for (auto& p : deviceInfoList_) {
      logger.handleDeviceInfo(p.first, p.second);
    }
    for (auto& p : resourceInfoList_) {
      logger.handleResourceInfo(p.first, p.second);
    }
    for (auto& cpu_trace_buffer : buffers_->cpu) {
      logger.handleTraceSpan(cpu_trace_buffer->span);
    }
    logger.finalizeTrace(*config_, nullptr, endTime_, loggerMetadata_);
  }

 private:
  std::unique_ptr<Config> config_;
  std::vector<const ITraceActivity*> activities_;
  std::vector<std::unique_ptr<const ITraceActivity>> wrappers_;
  std::vector<std::pair<DeviceInfo, int64_t>> deviceInfoList_;
  std::vector<std::pair<ResourceInfo, int64_t>> resourceInfoList_;
  std::unique_ptr<XPUActivityBuffers> buffers_;
  std::unordered_map<std::string, std::string> metadata_;
  std::unordered_map<std::string, std::vector<std::string>> loggerMetadata_;
  int64_t endTime_{0};
};

} // namespace KINETO_NAMESPACE
