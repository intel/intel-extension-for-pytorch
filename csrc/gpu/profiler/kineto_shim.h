#pragma once

#include <torch/csrc/profiler/api.h>

#include <set>

namespace torch_ipex::xpu {
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
} // namespace torch_ipex::xpu
