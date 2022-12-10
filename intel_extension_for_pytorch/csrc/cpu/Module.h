#pragma once

#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace torch_ipex {
// Initialize bindings for IPE module, tensor and optimization passes.
TORCH_API void init_cpu_module(py::module m);
} // namespace torch_ipex
