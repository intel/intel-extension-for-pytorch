#pragma once

#include <profiler/XPUActivityApi.h>
#include <profiler/XPUActivityPlatform.h>
#include <profiler/include/kineto/GenericTraceActivity.h>
#include <profiler/include/kineto/ITraceActivity.h>
#include <profiler/include/kineto/ThreadUtil.h>

#include <pti/pti_view.h>

#include <string>
#include <vector>

namespace libkineto {
class ActivityLogger;
}

namespace KINETO_NAMESPACE {

using namespace libkineto;
struct TraceSpan;
// This vector contains the legal runtime ops for drawing the correlation line
// in the output json file
inline const std::vector<std::string> correlate_runtime_ops = {
    "piextUSMEnqueueFill",
    "piextUSMEnqueueFill2D",
    "piextUSMEnqueueMemcpy",
    "piextUSMEnqueueMemset",
    "piextUSMEnqueueMemcpy2D",
    "piextUSMEnqueueMemset2D",
    "piEnqueueKernelLaunch",
    "piextEnqueueKernelLaunchCustom",
    "piextEnqueueCooperativeKernelLaunch"};

template <class T>
struct XPUActivity : public ITraceActivity {
  explicit XPUActivity(const T* activity, const ITraceActivity* linked)
      : activity_(*activity), linked_(linked) {}
  int64_t timestamp() const override;
  int64_t duration() const override;
  int64_t correlationId() const override {
    return 0;
  }
  int32_t getThreadId() const override {
    return 0;
  }
  const ITraceActivity* linkedActivity() const override {
    return linked_;
  }
  int flowType() const override {
    return kLinkAsyncCpuGpu;
  }
  int flowId() const override {
    return correlationId();
  }
  const T& raw() const {
    return activity_;
  }
  const TraceSpan* traceSpan() const override {
    return nullptr;
  }

 protected:
  const T& activity_;
  const ITraceActivity* linked_{nullptr};
};

// Onepti_ActivityAPI - ONEPTI runtime activities
struct RuntimeActivity : public XPUActivity<pti_view_record_sycl_runtime> {
  explicit RuntimeActivity(
      const pti_view_record_sycl_runtime* activity,
      const ITraceActivity* linked,
      int32_t threadId)
      : XPUActivity(activity, linked), threadId_(threadId) {}
  int64_t correlationId() const override {
    return activity_._correlation_id;
  }
  int64_t deviceId() const override {
    return processId();
  }
  int64_t resourceId() const override {
    return threadId_;
  }
  ActivityType type() const override {
    return ActivityType::XPU_RUNTIME;
  }
  bool flowStart() const override;
  const std::string name() const override {
    return activity_._name;
  }
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// Onepti_ActivityAPI - ONEPTI overhead activities
struct OverheadActivity : public XPUActivity<pti_view_record_overhead> {
  explicit OverheadActivity(
      const pti_view_record_overhead* activity,
      const ITraceActivity* linked,
      int32_t threadId = 0)
      : XPUActivity(activity, linked), threadId_(threadId) {}
  // TODO: Update this with PID ordering
  int64_t deviceId() const override {
    return -1;
  }
  int64_t resourceId() const override {
    return threadId_;
  }
  ActivityType type() const override {
    return ActivityType::OVERHEAD;
  }
  bool flowStart() const override;
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// Base class for GPU activities.
// Can also be instantiated directly.
template <class T>
struct GpuActivity : public XPUActivity<T> {
  explicit GpuActivity(const T* activity, const ITraceActivity* linked)
      : XPUActivity<T>(activity, linked) {}
  int64_t correlationId() const override {
    return raw()._correlation_id;
  }
  int64_t deviceId() const override {
    return XPUActivityApi::singleton().get_device_idx_from_uuid(
        raw()._device_uuid);
  }
  int64_t resourceId() const override {
    return (int64_t)raw()._queue_handle;
  }
  ActivityType type() const override;
  bool flowStart() const override {
    return false;
  }
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  const T& raw() const {
    return XPUActivity<T>::raw();
  }
};

} // namespace KINETO_NAMESPACE
