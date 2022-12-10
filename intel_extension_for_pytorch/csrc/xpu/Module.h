#pragma once

#include <pybind11/pybind11.h>

namespace xpu {

void init_xpu_module(pybind11::module& m);

} // namespace xpu
