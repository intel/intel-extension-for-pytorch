#include "cpu/init_python_bindings.h"
#include "xpu/init_python_bindings.h"

PYBIND11_MODULE(_C, m) {
  torch_ipex::InitIpexCpuBindings(m);
  xpu::InitIpexXpuBindings(m);
}
