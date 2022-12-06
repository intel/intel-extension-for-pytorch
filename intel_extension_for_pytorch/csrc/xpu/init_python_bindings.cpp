#include "init_python_bindings.h"
#include <c10/macros/Macros.h>
#include "Module.h"

namespace py = pybind11;

namespace xpu {

TORCH_API void InitIpexXpuBindings(py::module& m) {
#ifdef BUILD_WITH_XPU
  init_xpu_module(m);
#endif
}

} // namespace xpu
