/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "kineto/Config.h"

#include <stdlib.h>

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <time.h>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <istream>
#include <mutex>
#include <ostream>
#include <sstream>

#include "Logger.h"
#include "kineto/ThreadUtil.h"

using namespace std::chrono;

using std::string;
using std::vector;

namespace KINETO_NAMESPACE {

constexpr std::chrono::milliseconds Config::kControllerIntervalMsecs;

constexpr milliseconds kDefaultSamplePeriodMsecs(1000);
constexpr milliseconds kDefaultMultiplexPeriodMsecs(1000);
constexpr milliseconds kDefaultActivitiesProfileDurationMSecs(500);
constexpr int kDefaultActivitiesMaxGpuBufferSize(128 * 1024 * 1024);
constexpr seconds kDefaultActivitiesWarmupDurationSecs(5);
constexpr seconds kDefaultReportPeriodSecs(1);
constexpr int kDefaultSamplesPerReport(1);
constexpr int kDefaultMaxEventProfilersPerGpu(1);
constexpr int kDefaultEventProfilerHearbeatMonitorPeriod(0);
constexpr seconds kMaxRequestAge(10);
constexpr seconds kDefaultOnDemandConfigUpdateIntervalSecs(5);
// 3200000 is the default value set by CUPTI
constexpr size_t kDefaultCuptiDeviceBufferSize(3200000);
// Default value set by CUPTI is 250
constexpr size_t kDefaultCuptiDeviceBufferPoolLimit(20);

// Event Profiler
constexpr char kEventsKey[] = "EVENTS";
constexpr char kMetricsKey[] = "METRICS";
constexpr char kSamplePeriodKey[] = "SAMPLE_PERIOD_MSECS";
constexpr char kMultiplexPeriodKey[] = "MULTIPLEX_PERIOD_MSECS";
constexpr char kReportPeriodKey[] = "REPORT_PERIOD_SECS";
constexpr char kSamplesPerReportKey[] = "SAMPLES_PER_REPORT";
constexpr char kEventsLogFileKey[] = "EVENTS_LOG_FILE";
constexpr char kEventsEnabledDevicesKey[] = "EVENTS_ENABLED_DEVICES";
constexpr char kOnDemandDurationKey[] = "EVENTS_DURATION_SECS";
constexpr char kMaxEventProfilersPerGpuKey[] = "MAX_EVENT_PROFILERS_PER_GPU";
constexpr char kHeartbeatMonitorPeriodKey[] =
    "EVENTS_HEARTBEAT_MONITOR_PERIOD_SECS";

// Activity Profiler
constexpr char kActivitiesEnabledKey[] = "ACTIVITIES_ENABLED";
constexpr char kActivityTypesKey[] = "ACTIVITY_TYPES";
constexpr char kActivitiesLogFileKey[] = "ACTIVITIES_LOG_FILE";
constexpr char kActivitiesDurationKey[] = "ACTIVITIES_DURATION_SECS";
constexpr char kActivitiesDurationMsecsKey[] = "ACTIVITIES_DURATION_MSECS";
constexpr char kActivitiesWarmupDurationSecsKey[] =
    "ACTIVITIES_WARMUP_PERIOD_SECS";
constexpr char kActivitiesMaxGpuBufferSizeKey[] =
    "ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB";
constexpr char kActivitiesDisplayCudaSyncWaitEvents[] =
    "ACTIVITIES_DISPLAY_CUDA_SYNC_WAIT_EVENTS";

// Client Interface
// TODO: keep supporting these older config options, deprecate in the future
// using replacements.
constexpr char kClientInterfaceEnableOpInputsCollection[] =
    "CLIENT_INTERFACE_ENABLE_OP_INPUTS_COLLECTION";
constexpr char kPythonStackTrace[] = "PYTHON_STACK_TRACE";
// Profiler Config Options
constexpr char kProfileReportInputShapes[] = "PROFILE_REPORT_INPUT_SHAPES";
constexpr char kProfileProfileMemory[] = "PROFILE_PROFILE_MEMORY";
constexpr char kProfileWithStack[] = "PROFILE_WITH_STACK";
constexpr char kProfileWithFlops[] = "PROFILE_WITH_FLOPS";
constexpr char kProfileWithModules[] = "PROFILE_WITH_MODULES";

constexpr char kActivitiesWarmupIterationsKey[] =
    "ACTIVITIES_WARMUP_ITERATIONS";
constexpr char kActivitiesIterationsKey[] = "ACTIVITIES_ITERATIONS";
// Common

// Client-side timestamp used for synchronized start across hosts for
// distributed workloads.
// Specified in milliseconds Unix time (milliseconds since epoch).
// To use, compute a future timestamp as follows:
//    * C++: <delay_ms> + duration_cast<milliseconds>(
//               system_clock::now().time_since_epoch()).count()
//    * Python: <delay_ms> + int(time.time() * 1000)
//    * Bash: $((<delay_ms> + $(date +%s%3N)))
//    * Bash: $(date -d "$time + <delay_secs>seconds" +%s%3N)
// If used for a tracing request, timestamp must be far enough in the future
// to accommodate ACTIVITIES_WARMUP_PERIOD_SECS as well as any delays in
// propagating the request to the profiler.
// If the request can not be honored, it is up to the profilers to report
// an error somehow - no checks are done at config parse time.
// Note PROFILE_START_ITERATION has higher precedence
constexpr char kProfileStartTimeKey[] = "PROFILE_START_TIME";
// DEPRECATED - USE PROFILE_START_TIME instead
constexpr char kRequestTimestampKey[] = "REQUEST_TIMESTAMP";

// Alternatively if the application supports reporting iterations
// start the profile at specific iteration. If the iteration count
// is >= this value the profile is started immediately.
// A value >= 0 is valid for this config option to take effect.
// Note PROFILE_START_ITERATION will take precedence over PROFILE_START_TIME.
constexpr char kProfileStartIterationKey[] = "PROFILE_START_ITERATION";

// Users can also start the profile on an integer multiple of the config
// value PROFILE_START_ITERATION_ROUNDUP. This knob behaves similar to
// PROFILE_START_ITERATION but instead of saying : "start collection trace on
// iteration 500", one can configure it to "start collecting trace on the next
// 100th iteration".
//
// For example,
//   PROFILE_START_ITERATION_ROUNDUP = 1000, and the current iteration is 2010
//   The profile will then be collected on the next multiple of 1000 ie. 3000
// Note PROFILE_START_ITERATION_ROUNDUP will also take precedence over
// PROFILE_START_TIME.
constexpr char kProfileStartIterationRoundUpKey[] =
    "PROFILE_START_ITERATION_ROUNDUP";

// Enable on-demand trigger via kill -USR2 <pid>
// When triggered in this way, /tmp/libkineto.conf will be used as config.
constexpr char kEnableSigUsr2Key[] = "ENABLE_SIGUSR2";

// Enable communication through IPC Fabric
// and disable thrift communication with dynolog daemon
constexpr char kEnableIpcFabricKey[] = "ENABLE_IPC_FABRIC";
// Period to pull on-demand config from dynolog daemon
constexpr char kOnDemandConfigUpdateIntervalSecsKey[] =
    "ON_DEMAND_CONFIG_UPDATE_INTERVAL_SECS";

// Verbose log level
// The actual glog is not used and --v and --vmodule has no effect.
// Instead set the verbose level and modules in the config file.
constexpr char kLogVerboseLevelKey[] = "VERBOSE_LOG_LEVEL";
// By default, all modules will log verbose messages >= verboseLogLevel.
// But to reduce noise we can specify one or more modules of interest.
// A module is a C/C++ object file (source file name),
// Example argument: ActivityProfiler.cpp,output_json.cpp
constexpr char kLogVerboseModulesKey[] = "VERBOSE_LOG_MODULES";

// Max devices supported on any system
constexpr uint8_t kMaxDevices = 8;

namespace {

struct FactoryMap {
  void addFactory(
      std::string name,
      std::function<AbstractConfig*(Config&)> factory) {
    std::lock_guard<std::mutex> lock(lock_);
    factories_[name] = factory;
  }

  void addFeatureConfigs(Config& cfg) {
    std::lock_guard<std::mutex> lock(lock_);
    for (const auto& p : factories_) {
      cfg.addFeature(p.first, p.second(cfg));
    }
  }

  // Config factories are shared between objects and since
  // config objects can be created by multiple threads, we need a lock.
  std::mutex lock_;
  std::map<std::string, std::function<AbstractConfig*(Config&)>> factories_;
};

std::shared_ptr<FactoryMap> configFactories() {
  // Ensure this is safe to call during shutdown, even as static
  // destructors are invoked. getStaticObjectLifetimeHandle hangs onto
  // FactoryMap delaying its destruction.
  static auto factories = std::make_shared<FactoryMap>();
  static std::weak_ptr<FactoryMap> weak_ptr = factories;
  return weak_ptr.lock();
}

} // namespace

void Config::addConfigFactory(
    std::string name,
    std::function<AbstractConfig*(Config&)> factory) {
  auto factories = configFactories();
  if (factories) {
    factories->addFactory(name, factory);
  }
}

static string defaultTraceFileName() {
  return fmt::format("/tmp/libkineto_activities_{}.json", processId());
}

Config::Config()
    : verboseLogLevel_(-1),
      samplePeriod_(kDefaultSamplePeriodMsecs),
      reportPeriod_(duration_cast<milliseconds>(kDefaultReportPeriodSecs)),
      samplesPerReport_(kDefaultSamplesPerReport),
      eventProfilerOnDemandDuration_(seconds(0)),
      eventProfilerMaxInstancesPerGpu_(kDefaultMaxEventProfilersPerGpu),
      eventProfilerHeartbeatMonitorPeriod_(
          kDefaultEventProfilerHearbeatMonitorPeriod),
      multiplexPeriod_(kDefaultMultiplexPeriodMsecs),
      activityProfilerEnabled_(true),
      activitiesLogFile_(defaultTraceFileName()),
      activitiesLogUrl_(fmt::format("file://{}", activitiesLogFile_)),
      activitiesMaxGpuBufferSize_(kDefaultActivitiesMaxGpuBufferSize),
      activitiesWarmupDuration_(kDefaultActivitiesWarmupDurationSecs),
      activitiesWarmupIterations_(0),
      activitiesCudaSyncWaitEvents_(true),
      activitiesDuration_(kDefaultActivitiesProfileDurationMSecs),
      activitiesRunIterations_(0),
      activitiesOnDemandTimestamp_(milliseconds(0)),
      profileStartTime_(milliseconds(0)),
      profileStartIteration_(-1),
      profileStartIterationRoundUp_(-1),
      requestTimestamp_(milliseconds(0)),
      enableSigUsr2_(false),
      enableIpcFabric_(false),
      onDemandConfigUpdateIntervalSecs_(
          kDefaultOnDemandConfigUpdateIntervalSecs),
      cuptiDeviceBufferSize_(kDefaultCuptiDeviceBufferSize),
      cuptiDeviceBufferPoolLimit_(kDefaultCuptiDeviceBufferPoolLimit) {
  auto factories = configFactories();
  if (factories) {
    factories->addFeatureConfigs(*this);
  }
#if __linux__
  enableIpcFabric_ = (getenv(kUseDaemonEnvVar) != nullptr);
#endif
}

std::shared_ptr<void> Config::getStaticObjectsLifetimeHandle() {
  return configFactories();
}

uint8_t Config::createDeviceMask(const string& val) {
  uint8_t res = 0;
  for (const auto& d : splitAndTrim(val, ',')) {
    res |= 1 << toIntRange(d, 0, kMaxDevices - 1);
  }
  return res;
}

const seconds Config::maxRequestAge() const {
  return kMaxRequestAge;
}

static std::string getTimeStr(time_point<system_clock> t) {
  std::time_t t_c = system_clock::to_time_t(t);
  return fmt::format("{:%H:%M:%S}", fmt::localtime(t_c));
}

static time_point<system_clock> handleRequestTimestamp(int64_t ms) {
  auto t = time_point<system_clock>(milliseconds(ms));
  auto now = system_clock::now();
  if (t > now) {
    throw std::invalid_argument(fmt::format(
        "Invalid {}: {} - time is in future",
        kRequestTimestampKey,
        getTimeStr(t)));
  } else if ((now - t) > kMaxRequestAge) {
    throw std::invalid_argument(fmt::format(
        "Invalid {}: {} - time is more than {}s in the past",
        kRequestTimestampKey,
        getTimeStr(t),
        kMaxRequestAge.count()));
  }
  return t;
}

static time_point<system_clock> handleProfileStartTime(int64_t start_time_ms) {
  // If 0, return 0, so that AbstractConfig::parse can fix the timestamp later.
  if (start_time_ms == 0) {
    return time_point<system_clock>(milliseconds(0));
  }

  auto t = time_point<system_clock>(milliseconds(start_time_ms));
  // This should check that ProfileStartTime is in the future with
  // enough time for warm-up.
  // Unfortunately, warm-up duration is unknown at this point.
  // But we can still check that the start time is not in the past.
  auto now = system_clock::now();
  if ((now - t) > kMaxRequestAge) {
    throw std::invalid_argument(fmt::format(
        "Invalid {}: {} - start time is more than {}s in the past",
        kProfileStartTimeKey,
        getTimeStr(t),
        kMaxRequestAge.count()));
  }
  return t;
}

void Config::setActivityTypes(
    const std::vector<std::string>& selected_activities) {
  selectedActivityTypes_.clear();
  if (selected_activities.size() > 0) {
    for (const auto& activity : selected_activities) {
      if (activity == "") {
        continue;
      }
      selectedActivityTypes_.insert(toActivityType(activity));
    }
  }
}

bool Config::handleOption(const std::string& name, std::string& val) {
  // Event Profiler
  if (!name.compare(kEventsKey)) {
    vector<string> event_names = splitAndTrim(val, ',');
    eventNames_.insert(event_names.begin(), event_names.end());
  } else if (!name.compare(kMetricsKey)) {
    vector<string> metric_names = splitAndTrim(val, ',');
    metricNames_.insert(metric_names.begin(), metric_names.end());
  } else if (!name.compare(kSamplePeriodKey)) {
    samplePeriod_ = milliseconds(toInt32(val));
  } else if (!name.compare(kMultiplexPeriodKey)) {
    multiplexPeriod_ = milliseconds(toInt32(val));
  } else if (!name.compare(kReportPeriodKey)) {
    setReportPeriod(seconds(toInt32(val)));
  } else if (!name.compare(kSamplesPerReportKey)) {
    samplesPerReport_ = toInt32(val);
  } else if (!name.compare(kEventsLogFileKey)) {
    eventLogFile_ = val;
  } else if (!name.compare(kEventsEnabledDevicesKey)) {
    eventProfilerDeviceMask_ = createDeviceMask(val);
  } else if (!name.compare(kOnDemandDurationKey)) {
    eventProfilerOnDemandDuration_ = seconds(toInt32(val));
    eventProfilerOnDemandTimestamp_ = timestamp();
  } else if (!name.compare(kMaxEventProfilersPerGpuKey)) {
    eventProfilerMaxInstancesPerGpu_ = toInt32(val);
  } else if (!name.compare(kHeartbeatMonitorPeriodKey)) {
    eventProfilerHeartbeatMonitorPeriod_ = seconds(toInt32(val));
  }

  // Activity Profiler
  else if (!name.compare(kActivitiesDurationKey)) {
    activitiesDuration_ = duration_cast<milliseconds>(seconds(toInt32(val)));
    activitiesOnDemandTimestamp_ = timestamp();
  } else if (!name.compare(kActivityTypesKey)) {
    vector<string> activity_types = splitAndTrim(toLower(val), ',');
    setActivityTypes(activity_types);
  } else if (!name.compare(kActivitiesDurationMsecsKey)) {
    activitiesDuration_ = milliseconds(toInt32(val));
    activitiesOnDemandTimestamp_ = timestamp();
  } else if (!name.compare(kActivitiesIterationsKey)) {
    activitiesRunIterations_ = toInt32(val);
    activitiesOnDemandTimestamp_ = timestamp();
  } else if (!name.compare(kLogVerboseLevelKey)) {
    verboseLogLevel_ = toInt32(val);
  } else if (!name.compare(kLogVerboseModulesKey)) {
    verboseLogModules_ = splitAndTrim(val, ',');
  } else if (!name.compare(kActivitiesEnabledKey)) {
    activityProfilerEnabled_ = toBool(val);
  } else if (!name.compare(kActivitiesLogFileKey)) {
    activitiesLogFile_ = val;
    activitiesLogUrl_ = fmt::format("file://{}", val);
    size_t jidx = activitiesLogUrl_.find(".pt.trace.json");
    if (jidx != std::string::npos) {
      activitiesLogUrl_.replace(
          jidx, 14, fmt::format("_{}.pt.trace.json", processId()));
    } else {
      jidx = activitiesLogUrl_.find(".json");
      if (jidx != std::string::npos) {
        activitiesLogUrl_.replace(
            jidx, 5, fmt::format("_{}.json", processId()));
      }
    }
    activitiesOnDemandTimestamp_ = timestamp();
  } else if (!name.compare(kActivitiesMaxGpuBufferSizeKey)) {
    activitiesMaxGpuBufferSize_ = toInt32(val) * 1024 * 1024;
  } else if (!name.compare(kActivitiesWarmupDurationSecsKey)) {
    activitiesWarmupDuration_ = seconds(toInt32(val));
  } else if (!name.compare(kActivitiesWarmupIterationsKey)) {
    activitiesWarmupIterations_ = toInt32(val);
  } else if (!name.compare(kActivitiesDisplayCudaSyncWaitEvents)) {
    activitiesCudaSyncWaitEvents_ = toBool(val);
  }

  // TODO: Deprecate Client Interface
  else if (!name.compare(kClientInterfaceEnableOpInputsCollection)) {
    enableReportInputShapes_ = toBool(val);
  } else if (!name.compare(kPythonStackTrace)) {
    enableWithStack_ = toBool(val);
  }

  // Profiler Config
  else if (!name.compare(kProfileReportInputShapes)) {
    enableReportInputShapes_ = toBool(val);
  } else if (!name.compare(kProfileProfileMemory)) {
    enableProfileMemory_ = toBool(val);
  } else if (!name.compare(kProfileWithStack)) {
    enableWithStack_ = toBool(val);
  } else if (!name.compare(kProfileWithFlops)) {
    enableWithFlops_ = toBool(val);
  } else if (!name.compare(kProfileWithModules)) {
    enableWithModules_ = toBool(val);
  }

  // Common
  else if (!name.compare(kRequestTimestampKey)) {
    LOG(INFO) << kRequestTimestampKey << " has been deprecated - please use "
              << kProfileStartTimeKey;
    requestTimestamp_ = handleRequestTimestamp(toInt64(val));
  } else if (!name.compare(kProfileStartTimeKey)) {
    profileStartTime_ = handleProfileStartTime(toInt64(val));
  } else if (!name.compare(kProfileStartIterationKey)) {
    profileStartIteration_ = toInt32(val);
  } else if (!name.compare(kProfileStartIterationRoundUpKey)) {
    profileStartIterationRoundUp_ = toInt32(val);
  } else if (!name.compare(kEnableSigUsr2Key)) {
    enableSigUsr2_ = toBool(val);
  } else if (!name.compare(kEnableIpcFabricKey)) {
    enableIpcFabric_ = toBool(val);
  } else if (!name.compare(kOnDemandConfigUpdateIntervalSecsKey)) {
    onDemandConfigUpdateIntervalSecs_ = seconds(toInt32(val));
  } else {
    return false;
  }
  return true;
}

void Config::updateActivityProfilerRequestReceivedTime() {
  activitiesOnDemandTimestamp_ = system_clock::now();
}

void Config::setClientDefaults() {
  AbstractConfig::setClientDefaults();
  activitiesLogToMemory_ = true;
}

void Config::validate(
    const time_point<system_clock>& fallbackProfileStartTime) {
  if (samplePeriod_.count() == 0) {
    LOG(WARNING) << "Sample period must be greater than 0, setting to 1ms";
    samplePeriod_ = milliseconds(1);
  }

  if (multiplexPeriod_ < samplePeriod_) {
    LOG(WARNING) << "Multiplex period can not be smaller "
                 << "than sample period";
    LOG(WARNING) << "Setting multiplex period to " << samplePeriod_.count()
                 << "ms";
    multiplexPeriod_ = samplePeriod_;
  }

  if ((multiplexPeriod_ % samplePeriod_).count() != 0) {
    LOG(WARNING) << "Multiplex period must be a "
                 << "multiple of sample period";
    multiplexPeriod_ = alignUp(multiplexPeriod_, samplePeriod_);
    LOG(WARNING) << "Setting multiplex period to " << multiplexPeriod_.count()
                 << "ms";
  }

  if ((reportPeriod_ % multiplexPeriod_).count() != 0 ||
      reportPeriod_.count() == 0) {
    LOG(WARNING) << "Report period must be a "
                 << "multiple of multiplex period";
    reportPeriod_ = alignUp(reportPeriod_, multiplexPeriod_);
    LOG(WARNING) << "Setting report period to " << reportPeriod_.count()
                 << "ms";
  }

  if (samplesPerReport_ < 1) {
    LOG(WARNING) << "Samples per report must be in the range "
                 << "[1, report period / sample period]";
    LOG(WARNING) << "Setting samples per report to 1";
    samplesPerReport_ = 1;
  }

  int max_samples_per_report = reportPeriod_ / samplePeriod_;
  if (samplesPerReport_ > max_samples_per_report) {
    LOG(WARNING) << "Samples per report must be in the range "
                 << "[1, report period / sample period] ([1, "
                 << reportPeriod_.count() << "ms / " << samplePeriod_.count()
                 << "ms = " << max_samples_per_report << "])";
    LOG(WARNING) << "Setting samples per report to " << max_samples_per_report;
    samplesPerReport_ = max_samples_per_report;
  }

  if (!hasProfileStartTime()) {
    VLOG(0)
        << "No explicit timestamp has been set. "
        << "Defaulting it to now + activitiesWarmupDuration with a buffer of double the period of the monitoring thread.";
    profileStartTime_ = fallbackProfileStartTime + activitiesWarmupDuration() +
        2 * Config::kControllerIntervalMsecs;
  }

  if (profileStartIterationRoundUp_ == 0) {
    // setting to 0 will mess up modulo arithmetic, set it to -1 so it has no
    // effect
    LOG(WARNING) << "Profiler start iteration round up should be >= 1.";
    profileStartIterationRoundUp_ = -1;
  }

  if (profileStartIterationRoundUp_ > 0 && !hasProfileStartIteration()) {
    VLOG(0) << "Setting profiler start iteration to 0 so this config is "
            << "triggered via iteration count.";
    profileStartIteration_ = 0;
  }

  if (selectedActivityTypes_.size() == 0) {
    selectDefaultActivityTypes();
  }
}

void Config::setReportPeriod(milliseconds msecs) {
  reportPeriod_ = msecs;
}

void Config::printActivityProfilerConfig(std::ostream& s) const {
  s << "  Log file: " << activitiesLogFile() << std::endl;
  if (hasProfileStartIteration()) {
    s << "  Trace start Iteration: " << profileStartIteration() << std::endl;
    s << "  Trace warmup Iterations: " << activitiesWarmupIterations()
      << std::endl;
    s << "  Trace profile Iterations: " << activitiesRunIterations()
      << std::endl;
    if (profileStartIterationRoundUp() > 0) {
      s << "  Trace start iteration roundup : "
        << profileStartIterationRoundUp() << std::endl;
    }
  } else if (hasProfileStartTime()) {
    std::time_t t_c = system_clock::to_time_t(requestTimestamp());
    s << "  Trace start time: "
      << fmt::format("{:%Y-%m-%d %H:%M:%S}", fmt::localtime(t_c));
    s << "  Trace duration: " << activitiesDuration().count() << "ms"
      << std::endl;
    s << "  Warmup duration: " << activitiesWarmupDuration().count() << "s"
      << std::endl;
  }

  s << "  Max GPU buffer size: " << activitiesMaxGpuBufferSize() / 1024 / 1024
    << "MB" << std::endl;

  std::vector<const char*> activities;
  for (const auto& activity : selectedActivityTypes_) {
    activities.push_back(toString(activity));
  }
  s << "  Enabled activities: " << fmt::format("{}", fmt::join(activities, ","))
    << std::endl;

  AbstractConfig::printActivityProfilerConfig(s);
}

} // namespace KINETO_NAMESPACE
