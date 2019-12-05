#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace torch {
namespace jit {

// Initialize bindings for IPE module, tensor and optimization passes.
void InitIpeBindings(py::module m);

}  // namespace jit
}  // namespace torch
