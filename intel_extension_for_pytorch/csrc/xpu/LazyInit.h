#pragma once

namespace xpu {
namespace dpcpp {

// C++ API for lazy initialization. Note: don't put this function in destructor,
// because it may result in a deadlock.
void lazy_init();

void set_run_yet_variable_to_false();

void set_run_yet_variable_to_true();

} // namespace dpcpp
} // namespace xpu
