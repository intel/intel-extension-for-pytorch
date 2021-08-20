#pragma once

#include <torch/csrc/python_headers.h>

PyObject* THPStorage_postInitExtension(PyObject* module);
PyObject* THPStorage_init(PyObject* module);
