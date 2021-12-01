#pragma once

#include <torch/csrc/python_headers.h>

PyObject* THDPStorage_postInitExtension(PyObject* module);
PyObject* THDPStorage_init(PyObject* module);
