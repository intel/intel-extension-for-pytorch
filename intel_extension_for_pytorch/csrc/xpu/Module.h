#pragma once

#include <pybind11/pybind11.h>

namespace torch_ipex::xpu {

void init_xpu_module(pybind11::module& m);

} // namespace torch_ipex::xpu
