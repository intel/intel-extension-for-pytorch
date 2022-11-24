#pragma once

#include <torch/csrc/python_headers.h>

namespace xpu {

PyObject* THDPStorage_postInitExtension(PyObject* module);
PyObject* THDPStorage_init(PyObject* module);

} // namespace xpu
