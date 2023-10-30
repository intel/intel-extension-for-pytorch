#ifdef USE_KINETO

#include <profiler/XPUActivityProfilerProxy.h>

#include <chrono>

#include <profiler/Logger.h>
#include <profiler/XPUActivityApi.h>
#include <profiler/XPUActivityProfilerController.h>
#include <profiler/include/kineto/Config.h>

namespace KINETO_NAMESPACE {

XPUActivityProfilerProxy::XPUActivityProfilerProxy(bool cpuOnly)
    : cpuOnly_(cpuOnly) {}

XPUActivityProfilerProxy::~XPUActivityProfilerProxy() {
  delete controller_;
};

void XPUActivityProfilerProxy::init() {
  if (!controller_)
    controller_ = new XPUActivityProfilerController(cpuOnly_);
};

bool XPUActivityProfilerProxy::isActive() {
  return controller_->isActive();
}

void XPUActivityProfilerProxy::scheduleTrace(const std::string& configStr) {
  Config config;
  config.parse(configStr);
  controller_->scheduleTrace(config);
}

void XPUActivityProfilerProxy::scheduleTrace(const Config& config) {
  controller_->scheduleTrace(config);
}

void XPUActivityProfilerProxy::prepareTrace(
    const std::set<ActivityType>& activityTypes,
    const std::string& configStr) {
  Config config;
  bool validate_required = true;

  if (!configStr.empty()) {
    if (!config.parse(configStr)) {
      LOG(WARNING) << "Failed to parse config : " << configStr;
    }
    validate_required = false;
  }

  // auto loaded_config = configLoader_.getConfString();
  // if (!loaded_config.empty()) {
  //   config.parse(loaded_config);
  // }

  config.setClientDefaults();
  config.setSelectedActivityTypes(activityTypes);

  if (validate_required) {
    config.validate(std::chrono::system_clock::now());
  }

  controller_->prepareTrace(config);
}

void XPUActivityProfilerProxy::startTrace() {
  controller_->startTrace();
}

std::unique_ptr<ActivityTraceInterface> XPUActivityProfilerProxy::stopTrace() {
  return controller_->stopTrace();
}

void XPUActivityProfilerProxy::step() {
  controller_->step();
}

void XPUActivityProfilerProxy::pushCorrelationId(uint64_t id) {
  XPUActivityApi::pushCorrelationID(
      id, XPUActivityApi::CorrelationFlowType::Default);
}

void XPUActivityProfilerProxy::popCorrelationId() {
  XPUActivityApi::popCorrelationID(
      XPUActivityApi::CorrelationFlowType::Default);
}

void XPUActivityProfilerProxy::pushUserCorrelationId(uint64_t id) {
  XPUActivityApi::pushCorrelationID(
      id, XPUActivityApi::CorrelationFlowType::User);
}

void XPUActivityProfilerProxy::popUserCorrelationId() {
  XPUActivityApi::popCorrelationID(XPUActivityApi::CorrelationFlowType::User);
}

void XPUActivityProfilerProxy::transferCpuTrace(
    std::unique_ptr<CpuTraceBuffer> traceBuffer) {
  controller_->transferCpuTrace(std::move(traceBuffer));
}

void XPUActivityProfilerProxy::recordThreadInfo() {
  controller_->recordThreadInfo();
}

void XPUActivityProfilerProxy::addMetadata(
    const std::string& key,
    const std::string& value) {
  controller_->addMetadata(key, value);
}

void XPUActivityProfilerProxy::addChildActivityProfiler(
    std::unique_ptr<IActivityProfiler> profiler) {
  controller_->addChildActivityProfiler(std::move(profiler));
}

void XPUActivityProfilerProxy::logInvariantViolation(
    const std::string& profile_id,
    const std::string& assertion,
    const std::string& error,
    const std::string& group_profile_id) {
  controller_->logInvariantViolation(
      profile_id, assertion, error, group_profile_id);
}

} // namespace KINETO_NAMESPACE

#endif
