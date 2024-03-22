#pragma once

#include <torch/csrc/profiler/api.h>

#include <utils/Macros.h>

namespace torch_ipex::xpu {
namespace dpcpp {
namespace profiler {

IPEX_API void prepareProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities);

} // namespace profiler
} // namespace dpcpp
} // namespace torch_ipex::xpu
