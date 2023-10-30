#include <profiler/profiler_kineto.h>

#include <ATen/Context.h>
#include <torch/csrc/profiler/events.h>

#include <profiler/kineto_shim.h>

#ifdef USE_ONETRACE
#include <profiler/XPUActivityApi.h>
#include <runtime/Device.h>
#endif

namespace xpu {
namespace dpcpp {
namespace profiler {

using namespace torch::autograd::profiler;

void enableTracingLayer() {
#ifdef USE_ONETRACE
  setenv("ZE_ENABLE_TRACING_LAYER", "1", 1);
  libkineto::XPUActivityApi::singleton();
#endif
}

void prepareDevicePool() {
#ifdef USE_ONETRACE
  int device_count = 0;
  dpcppGetDeviceCount(&device_count);
  std::vector<std::string> devices;
  for (int device_index = 0; device_index < device_count; device_index++) {
    auto device = dpcppGetRawDevice((int8_t)device_index);
    auto device_handler =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
    std::stringstream ss;
    ss << std::hex << device_handler;
    std::string device_handler_str;
    ss >> device_handler_str;
    devices.push_back(device_handler_str);
  }
  libkineto::XPUActivityApi::singleton().setDeviceIdMap(devices);
#endif
}

void prepareProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities) {
  if (config.state == ProfilerState::NVTX ||
      config.state == ProfilerState::ITT) {
    return;
  }
  TORCH_CHECK(
      config.state == ProfilerState::KINETO ||
          config.state == ProfilerState::KINETO_GPU_FALLBACK,
      "Supported only in Kineto profiler");
  xpu::dpcpp::profiler::impl::kineto::prepareTrace(
      /*cpuOnly=*/!at::hasXPU(), activities, config.experimental_config);

#ifdef USE_ONETRACE
  prepareDevicePool();
#endif

  if (!config.experimental_config.performance_events.empty()) {
    TORCH_CHECK(
        activities.count(torch::autograd::profiler::ActivityType::CPU),
        "Cannot run cpu hardware profiler without CPU activities, please only use CPU activity type");
    auto is_standard_event = [](const std::string& event) -> bool {
      for (auto e : torch::profiler::ProfilerPerfEvents) {
        if (!std::strcmp(event.c_str(), e)) {
          return true;
        }
      }
      return false;
    };

    for (const auto& e : config.experimental_config.performance_events) {
      if (!is_standard_event(e)) {
        TORCH_WARN("Forwarding a non-stadard CPU performance event : ", e);
      }
    }
  }
}

} // namespace profiler
} // namespace dpcpp
} // namespace xpu
