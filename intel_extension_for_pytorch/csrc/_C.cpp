#include "cpu/init_python_bindings.h"

#ifdef BUILD_WITH_XPU
#include "xpu/init_python_bindings.h"
#endif

PYBIND11_MODULE(_C, m) {
  torch_ipex::InitIpexCpuBindings(m);

#ifdef BUILD_WITH_XPU
  xpu::InitIpexXpuBindings(m);
#endif
}
