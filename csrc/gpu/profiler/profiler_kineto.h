#pragma once

#include <torch/csrc/profiler/api.h>

#include <utils/Macros.h>

namespace xpu {
namespace dpcpp {
namespace profiler {

IPEX_API void enableTracingLayer();

IPEX_API void prepareProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities);

} // namespace profiler
} // namespace dpcpp
} // namespace xpu
