#pragma once

#include <pybind11/pybind11.h>

namespace xpu {

void InitIpexXpuBindings(pybind11::module& m);
}
