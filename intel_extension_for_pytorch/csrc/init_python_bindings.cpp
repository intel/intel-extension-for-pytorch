#include "init_python_bindings.h"
#include "cpu/Module.h"
#ifdef BUILD_WITH_XPU
#include "xpu/Module.h"
#endif

namespace py = pybind11;

void InitIpexBindings(py::module& m) {
#ifdef BUILD_WITH_CPU
  torch_ipex::init_cpu_module(m);
  m.def("_has_cpu", []() { return true; });
#else
  m.def("_has_cpu", []() { return false; });
#endif
#ifdef BUILD_WITH_XPU
  xpu::init_xpu_module(m);
  m.def("_has_xpu", []() { return true; });
#else
  m.def("_has_xpu", []() { return false; });
#endif

  // For IPEX CPU, the SynGraph availability is always false
  m.def("_is_syngraph_available", []() { return false; });
}
