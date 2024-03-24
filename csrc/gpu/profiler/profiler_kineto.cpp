#include <profiler/profiler_kineto.h>

#include <ATen/Context.h>
#include <torch/csrc/profiler/events.h>

#include <profiler/kineto_shim.h>

#ifdef USE_PTI
#include <ATen/xpu/XPUContext.h>
#include <profiler/XPUActivityApi.h>
#endif

namespace torch_ipex::xpu {
namespace dpcpp {
namespace profiler {

using namespace torch::autograd::profiler;

void prepareDevicePool() {
#ifdef USE_PTI
  // enum and save device uuids for PTI mapping
  auto device_count = at::xpu::device_count();
  std::vector<std::array<unsigned char, 16>> uuids;
  for (int device_index = 0; device_index < device_count; device_index++) {
    auto device = at::xpu::get_raw_device(device_index);
    if (device.is_gpu() &&
        device.has(sycl::aspect::ext_intel_device_info_uuid)) {
      uuids.push_back(device.get_info<sycl::ext::intel::info::device::uuid>());
    } else {
      uuids.push_back(std::array<unsigned char, 16>{});
    }
  }
  libkineto::XPUActivityApi::singleton().setDeviceUuidMap(uuids);
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
  torch_ipex::xpu::dpcpp::profiler::impl::kineto::prepareTrace(
      /*cpuOnly=*/!at::hasXPU(), activities, config.experimental_config);

#ifdef USE_PTI
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
} // namespace torch_ipex::xpu
