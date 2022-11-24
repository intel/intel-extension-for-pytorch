#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace xpu {
void InitIpexXpuBindings(py::module& m);
}