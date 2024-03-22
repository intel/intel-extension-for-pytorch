#include <profiler/kineto_shim.h>

#include <torch/csrc/profiler/api.h>

#ifdef USE_PTI
#include <profiler/XPUActivityProfilerProxy.h>
#include <profiler/include/kineto/libkineto.h>

namespace KINETO_NAMESPACE {
class XPUActivityProfilerProxy;
}
#endif

namespace torch_ipex::xpu {
namespace dpcpp {
namespace profiler {
namespace impl {
namespace kineto {

#ifdef USE_PTI
namespace {
const std::set<libkineto::ActivityType> cpuTypes = {
    libkineto::ActivityType::CPU_OP,
    libkineto::ActivityType::CPU_INSTANT_EVENT,
    libkineto::ActivityType::USER_ANNOTATION,
    libkineto::ActivityType::EXTERNAL_CORRELATION,
    libkineto::ActivityType::CUDA_RUNTIME,
    libkineto::ActivityType::XPU_RUNTIME,
    libkineto::ActivityType::PYTHON_FUNCTION,
};

const std::set<libkineto::ActivityType> xpuTypes = {
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::CONCURRENT_KERNEL,
    libkineto::ActivityType::XPU_RUNTIME,
};
} // namespace

void libkineto_init_xpu(bool cpuOnly, bool logOnError) {
  // libkineto::ConfigLoader& config_loader = libkineto::api().configLoader();
  libkineto::api().registerProfiler(
      std::make_unique<KINETO_NAMESPACE::XPUActivityProfilerProxy>(cpuOnly));
}
#endif

void prepareTrace(
    const bool cpuOnly,
    const ActivitySet& activities,
    const torch::profiler::impl::ExperimentalConfig& config) {
#ifdef USE_PTI
  if (!libkineto::api().isProfilerRegistered()) {
    libkineto_init_xpu(/*cpuOnly=*/cpuOnly, /*logOnError=*/true);
    libkineto::api().suppressLogMessages();
  }

  if (!libkineto::api().isProfilerInitialized()) {
    libkineto::api().initProfilerIfRegistered();
  }

  std::set<libkineto::ActivityType> k_activities;
  if (activities.count(torch::autograd::profiler::ActivityType::CPU)) {
    k_activities.insert(cpuTypes.begin(), cpuTypes.end());
  }
  if (activities.count(torch::autograd::profiler::ActivityType::XPU)) {
    k_activities.insert(xpuTypes.begin(), xpuTypes.end());
  }

  libkineto::api().activityProfiler().prepareTrace(k_activities);
#else
  TORCH_CHECK(
      false,
      "PTI support is not built. Cannot profile on XPU without PTI support");
#endif
}

} // namespace kineto
} // namespace impl
} // namespace profiler
} // namespace dpcpp
} // namespace torch_ipex::xpu
