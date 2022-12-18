#include "init_python_bindings.h"
#include "cpu/Module.h"
#ifdef BUILD_WITH_XPU
#include "xpu/Module.h"
#endif

namespace py = pybind11;

void InitIpexBindings(py::module& m) {
#ifdef BUILD_WITH_CPU
  torch_ipex::init_cpu_module(m);
#endif
#ifdef BUILD_WITH_XPU
  xpu::init_xpu_module(m);
#endif
}
