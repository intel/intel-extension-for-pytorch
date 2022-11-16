#include "init_python_bindings_xpu.h"

#include "Module.h"

namespace py = pybind11;
namespace xpu {

static std::vector<PyMethodDef> methods;

void InitIpexXpuBindings(py::module& m) {
  m.doc() = "PyTorch Extension for Intel XPU";
  init_xpu_module(m);
}
} // namespace xpu
