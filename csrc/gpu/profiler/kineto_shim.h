#pragma once

#include <torch/csrc/profiler/api.h>

#include <set>

namespace xpu {
namespace dpcpp {
namespace profiler {
namespace impl {
namespace kineto {

using ActivitySet = std::set<torch::autograd::profiler::ActivityType>;
void prepareTrace(
    const bool cpuOnly,
    const ActivitySet& activities,
    const torch::profiler::impl::ExperimentalConfig& config);

} // namespace kineto
} // namespace impl
} // namespace profiler
} // namespace dpcpp
} // namespace xpu
