#pragma once

#include <c10/macros/Macros.h>
#include <pybind11/pybind11.h>

TORCH_API void InitIpexBindings(pybind11::module& m);
