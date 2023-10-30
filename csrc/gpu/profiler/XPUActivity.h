#pragma once

#include <profiler/XPUActivityPlatform.h>
#include <profiler/include/kineto/GenericTraceActivity.h>
#include <profiler/include/kineto/ITraceActivity.h>
#include <profiler/include/kineto/ThreadUtil.h>
#include "onepti_activity_api.h"

#include <string>

namespace libkineto {
class ActivityLogger;
}

namespace KINETO_NAMESPACE {

using namespace libkineto;
struct TraceSpan;

template <class T>
struct XPUActivity : public ITraceActivity {
  explicit XPUActivity(const T* activity, const ITraceActivity* linked)
      : activity_(*activity), linked_(linked) {}
  int64_t timestamp() const override {
    return nsToUs(unixEpochTimestamp(activity_.start));
  }
  int64_t duration() const override {
    return nsToUs(activity_.end - activity_.start);
  }
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
struct RuntimeActivity : public XPUActivity<Onepti_ActivityAPI> {
  explicit RuntimeActivity(
      const Onepti_ActivityAPI* activity,
      const ITraceActivity* linked,
      int32_t threadId)
      : XPUActivity(activity, linked), threadId_(threadId) {}
  int64_t correlationId() const override {
    return activity_.correlationId;
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
    return activity_.name;
  }
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// // Onepti_ActivityAPI - ONEPTI overhead activities
// struct OverheadActivity : public XPUActivity<Onepti_ActivityOverhead> {
//   explicit OverheadActivity(
//       const Onepti_ActivityOverhead* activity,
//       const ITraceActivity* linked,
//       int32_t threadId=0)
//       : XPUActivity(activity, linked), threadId_(threadId) {}
//
//   int64_t timestamp() const override {
//     return nsToUs(unixEpochTimestamp(activity_.start));
//   }
//   int64_t duration() const override {
//     return nsToUs(activity_.end - activity_.start);
//   }
//   // TODO: Update this with PID ordering
//   int64_t deviceId() const override {return -1;}
//   int64_t resourceId() const override {return threadId_;}
//   ActivityType type() const override {return ActivityType::OVERHEAD;}
//   bool flowStart() const override;
//   const std::string name() const override {return
//   overheadKindString(activity_.overheadKind);} void log(ActivityLogger&
//   logger) const override; const std::string metadataJson() const override;
//
//  private:
//   const int32_t threadId_;
// };

// Base class for GPU activities.
// Can also be instantiated directly.
template <class T>
struct GpuActivity : public XPUActivity<T> {
  explicit GpuActivity(const T* activity, const ITraceActivity* linked)
      : XPUActivity<T>(activity, linked) {}
  int64_t correlationId() const override {
    return raw().correlationId;
  }
  int64_t deviceId() const override {
    return raw().deviceId;
  }
  int64_t resourceId() const override {
    return raw().queueId;
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
