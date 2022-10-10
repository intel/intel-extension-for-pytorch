#include "py_init.h"

#include "gpu/Module.h"

namespace py = pybind11;

static std::vector<PyMethodDef> methods;

void ipex_init(pybind11::module& m) {
  m.doc() = "PyTorch Extension for Intel XPU";
  init_module(m);
}
