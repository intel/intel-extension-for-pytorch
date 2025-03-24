#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "cpu/isa_help/isa_help.h"

namespace py = pybind11;

void InitIsaHelpModuleBindings(py::module m) {
  m.def("_check_isa_avx2", []() { return isa_help::check_isa_avx2(); });

  m.def("_check_isa_avx512", []() { return isa_help::check_isa_avx512(); });
}

PYBIND11_MODULE(_isa_help, m) {
  InitIsaHelpModuleBindings(m);
}
