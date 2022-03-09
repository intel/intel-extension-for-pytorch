#include "py_init.h"

#ifdef USE_ITT
#include <itt/itt.h>
#endif

#include "gpu/Module.h"

namespace py = pybind11;

static std::vector<PyMethodDef> methods;

void ipex_init(pybind11::module& m) {
  m.doc() = "PyTorch Extension for Intel XPU";
  init_module(m);

#if defined(USE_ITT)
  m.def("_itt_is_enabled", []() { return true; });
#else
  m.def("_itt_is_enabled", []() { return false; });
#endif

#ifdef USE_ITT
  auto module = m.ptr();
  IttBindings_init(module);
#endif
}
