#ifdef USE_KINETO

#include <profiler/XPUActivityProfilerController.h>

#include <chrono>
#include <functional>
#include <thread>

#include <profiler/ActivityLoggerFactory.h>
#include <profiler/Logger.h>
#include <profiler/XPUActivityApi.h>
#include <profiler/XPUActivityTrace.h>
#include <profiler/include/kineto/ThreadUtil.h>
#include <profiler/xpu_output_json.h>
#include <profiler/xpu_output_membuf.h>

using namespace std::chrono;

namespace KINETO_NAMESPACE {

constexpr milliseconds kProfilerIntervalMsecs(1000);

static std::unique_ptr<LoggerCollector>& loggerCollectorFactory() {
  static std::unique_ptr<LoggerCollector> factory = nullptr;
  return factory;
}

void XPUActivityProfilerController::setLoggerCollectorFactory(
    std::function<std::unique_ptr<LoggerCollector>()> factory) {
  loggerCollectorFactory() = factory();
}

XPUActivityProfilerController::XPUActivityProfilerController(bool cpuOnly) {
  profiler_ = std::make_unique<XPUActivityProfiler>(
      XPUActivityApi::singleton(), cpuOnly);

  if (loggerCollectorFactory()) {
    Logger::addLoggerObserver(loggerCollectorFactory().get());
  }
}

XPUActivityProfilerController::~XPUActivityProfilerController() {
  if (profilerThread_) {
    stopRunloop_ = true;
    profilerThread_->join();
    delete profilerThread_;
    profilerThread_ = nullptr;
  }

  if (loggerCollectorFactory()) {
    Logger::removeLoggerObserver(loggerCollectorFactory().get());
  }
}

static ActivityLoggerFactory initLoggerFactory() {
  ActivityLoggerFactory factory;
  factory.addProtocol("file", [](const std::string& url) {
    return std::unique_ptr<ActivityLogger>(new ChromeTraceLogger(url));
  });
  return factory;
}

static ActivityLoggerFactory& loggerFactory() {
  static ActivityLoggerFactory factory = initLoggerFactory();
  return factory;
}

void XPUActivityProfilerController::addLoggerFactory(
    const std::string& protocol,
    ActivityLoggerFactory::FactoryFunc factory) {
  loggerFactory().addProtocol(protocol, factory);
}

static std::unique_ptr<ActivityLogger> makeLogger(const Config& config) {
  if (config.activitiesLogToMemory()) {
    return std::make_unique<XPUMemoryTraceLogger>(config);
  }
  return loggerFactory().makeLogger(config.activitiesLogUrl());
}

static std::unique_ptr<InvariantViolationsLogger>&
invariantViolationsLoggerFactory() {
  static std::unique_ptr<InvariantViolationsLogger> factory = nullptr;
  return factory;
}

void XPUActivityProfilerController::setInvariantViolationsLoggerFactory(
    const std::function<std::unique_ptr<InvariantViolationsLogger>()>&
        factory) {
  invariantViolationsLoggerFactory() = factory();
}

bool XPUActivityProfilerController::canAcceptConfig() {
  return !profiler_->isActive();
}

void XPUActivityProfilerController::acceptConfig(const Config& config) {
  VLOG(1) << "acceptConfig";
  if (config.activityProfilerEnabled()) {
    scheduleTrace(config);
  }
}

bool XPUActivityProfilerController::shouldActivateTimestampConfig(
    const std::chrono::time_point<std::chrono::system_clock>& now) {
  if (asyncRequestConfig_->hasProfileStartIteration()) {
    return false;
  }
  if (now + kProfilerIntervalMsecs >=
      (asyncRequestConfig_->requestTimestamp() -
       asyncRequestConfig_->activitiesWarmupDuration())) {
    LOG(INFO)
        << "Received on-demand activity trace request by "
        << " profile timestamp = "
        << asyncRequestConfig_->requestTimestamp().time_since_epoch().count();
    return true;
  }
  return false;
}

bool XPUActivityProfilerController::shouldActivateIterationConfig(
    int64_t currentIter) {
  if (!asyncRequestConfig_->hasProfileStartIteration()) {
    return false;
  }
  auto rootIter = asyncRequestConfig_->startIterationIncludingWarmup();
  if (currentIter < rootIter) {
    return false;
  }

  LOG(INFO) << "Received on-demand activity trace request by "
            << " profile start iteration = "
            << asyncRequestConfig_->profileStartIteration()
            << ", current iteration = " << currentIter;
  if (currentIter > rootIter) {
    auto newProfileStart =
        currentIter + asyncRequestConfig_->activitiesWarmupIterations();
    if (asyncRequestConfig_->profileStartIterationRoundUp() > 0) {
      auto divisor = asyncRequestConfig_->profileStartIterationRoundUp();
      auto rem = newProfileStart % divisor;
      newProfileStart += ((rem == 0) ? 0 : divisor - rem);
      LOG(INFO) << "Rounding up profiler start iteration to : "
                << newProfileStart;
      asyncRequestConfig_->setProfileStartIteration(newProfileStart);
      if (currentIter != asyncRequestConfig_->startIterationIncludingWarmup()) {
        return false;
      }
    } else {
      LOG(INFO) << "Start iteration updated to : " << newProfileStart;
      asyncRequestConfig_->setProfileStartIteration(newProfileStart);
    }
  }
  return true;
}

void XPUActivityProfilerController::profilerLoop() {
  setThreadName("Kineto Activity Profiler");
  VLOG(0) << "Entering activity profiler loop";

  auto now = system_clock::now();
  auto next_wakeup_time = now + kProfilerIntervalMsecs;

  while (!stopRunloop_) {
    now = system_clock::now();

    while (now < next_wakeup_time) {
      std::this_thread::sleep_for(next_wakeup_time - now);
      now = system_clock::now();
    }

    if (asyncRequestConfig_ && !profiler_->isActive()) {
      std::lock_guard<std::mutex> lock(asyncConfigLock_);
      if (asyncRequestConfig_ && !profiler_->isActive() &&
          shouldActivateTimestampConfig(now)) {
        activateConfig(now);
      }
    }

    while (next_wakeup_time < now) {
      next_wakeup_time += kProfilerIntervalMsecs;
    }

    if (profiler_->isActive()) {
      next_wakeup_time = profiler_->performRunLoopStep(now, next_wakeup_time);
      VLOG(1) << "Profiler loop: "
              << duration_cast<milliseconds>(system_clock::now() - now).count()
              << "ms";
    }
  }

  VLOG(0) << "Exited activity profiling loop";
}

void XPUActivityProfilerController::step() {
  int64_t currentIter = ++iterationCount_;
  VLOG(0) << "Step called , iteration = " << currentIter;

  if (asyncRequestConfig_ && !profiler_->isActive()) {
    std::lock_guard<std::mutex> lock(asyncConfigLock_);
    auto now = system_clock::now();
    if (asyncRequestConfig_ && !profiler_->isActive() &&
        shouldActivateIterationConfig(currentIter)) {
      activateConfig(now);
    }
  }

  if (profiler_->isActive()) {
    auto now = system_clock::now();
    auto next_wakeup_time = now + kProfilerIntervalMsecs;
    profiler_->performRunLoopStep(now, next_wakeup_time, currentIter);
  }
}

void XPUActivityProfilerController::activateConfig(
    std::chrono::time_point<std::chrono::system_clock> now) {
  logger_ = makeLogger(*asyncRequestConfig_);
  profiler_->setLogger(logger_.get());
  LOGGER_OBSERVER_SET_TRIGGER_ON_DEMAND();
  profiler_->configure(*asyncRequestConfig_, now);
  asyncRequestConfig_ = nullptr;
}

void XPUActivityProfilerController::scheduleTrace(const Config& config) {
  VLOG(1) << "scheduleTrace";
  if (profiler_->isActive()) {
    LOG(WARNING) << "Ignored request - profiler busy";
    return;
  }
  int64_t currentIter = iterationCount_;
  if (config.hasProfileStartIteration() && currentIter < 0) {
    LOG(WARNING) << "Ignored profile iteration count based request as "
                 << "application is not updating iteration count";
    return;
  }

  bool newConfigScheduled = false;
  if (!asyncRequestConfig_) {
    std::lock_guard<std::mutex> lock(asyncConfigLock_);
    if (!asyncRequestConfig_) {
      asyncRequestConfig_ = config.clone();
      newConfigScheduled = true;
    }
  }
  if (!newConfigScheduled) {
    LOG(WARNING) << "Ignored request - another profile request is pending.";
    return;
  }

  if (!profilerThread_) {
    profilerThread_ =
        new std::thread(&XPUActivityProfilerController::profilerLoop, this);
  }
}

void XPUActivityProfilerController::prepareTrace(const Config& config) {
  auto now = system_clock::now();
  if (profiler_->isActive()) {
    LOG(WARNING) << "Cancelling current trace request in order to start "
                 << "higher priority synchronous request";
    if (libkineto::api().client()) {
      libkineto::api().client()->stop();
    }
    profiler_->stopTrace(now);
    profiler_->reset();
  }

  profiler_->configure(config, now);
}

void XPUActivityProfilerController::startTrace() {
  UST_LOGGER_MARK_COMPLETED(kWarmUpStage);
  profiler_->startTrace(std::chrono::system_clock::now());
}

std::unique_ptr<ActivityTraceInterface> XPUActivityProfilerController::
    stopTrace() {
  profiler_->stopTrace(std::chrono::system_clock::now());
  UST_LOGGER_MARK_COMPLETED(kCollectionStage);
  auto logger = std::make_unique<XPUMemoryTraceLogger>(profiler_->config());
  profiler_->processTrace(*logger);
  UST_LOGGER_MARK_COMPLETED(kPostProcessingStage);
  profiler_->reset();
  return std::make_unique<XPUActivityTrace>(std::move(logger), loggerFactory());
}

void XPUActivityProfilerController::addMetadata(
    const std::string& key,
    const std::string& value) {
  profiler_->addMetadata(key, value);
}

void XPUActivityProfilerController::logInvariantViolation(
    const std::string& profile_id,
    const std::string& assertion,
    const std::string& error,
    const std::string& group_profile_id) {
  if (invariantViolationsLoggerFactory()) {
    invariantViolationsLoggerFactory()->logInvariantViolation(
        profile_id, assertion, error, group_profile_id);
  }
}

} // namespace KINETO_NAMESPACE

#endif
