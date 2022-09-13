#pragma once

#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace torch_ipex {

// Initialize bindings for IPE module, tensor and optimization passes.
void InitIpexBindings(py::module m);

} // namespace torch_ipex
