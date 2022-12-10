#include "init_python_bindings.h"
#include "cpu/Module.h"
#ifdef BUILD_WITH_XPU
#include "xpu/Module.h"
#endif

namespace py = pybind11;

void InitIpexBindings(py::module& m) {
  torch_ipex::init_cpu_module(m);
#ifdef BUILD_WITH_XPU
  xpu::init_xpu_module(m);
#endif
}
