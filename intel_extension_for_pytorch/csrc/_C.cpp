#include "init_python_bindings.h"

PYBIND11_MODULE(_C, m) {
  InitIpexBindings(m);
}
