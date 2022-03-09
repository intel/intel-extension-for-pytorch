#include "py_init.h"

PYBIND11_MODULE(_C, m) {
  ipex_init(m);
}
