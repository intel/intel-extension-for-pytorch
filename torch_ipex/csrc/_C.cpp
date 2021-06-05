#include "py_init.h"

PYBIND11_MODULE(_C, m) {
  torch_ipex::InitIpexBindings(m);
}