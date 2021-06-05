#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace torch_ipex {

// Initialize bindings for IPE module, tensor and optimization passes.
void InitIpexBindings(py::module m);

}  // namespace torch_ipex