#pragma once

#include <Python.h>
#include <vector>

namespace xpu {

static inline void module_add_py_defs(
    std::vector<PyMethodDef>& vector,
    PyMethodDef* methods) {
  if (!vector.empty()) {
    // remove nullptr terminator
    vector.pop_back();
  }
  while (true) {
    vector.push_back(*methods);
    if (!methods->ml_name) {
      break;
    }
    methods++;
  }
}
} // namespace xpu
