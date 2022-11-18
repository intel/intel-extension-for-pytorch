#include "init_python_bindings.h"
#include "Module.h"

namespace py = pybind11;

namespace xpu {
static std::vector<PyMethodDef> methods;

void InitIpexXpuBindings(py::module& m) {
  init_xpu_module(m);
}
} // namespace xpu
