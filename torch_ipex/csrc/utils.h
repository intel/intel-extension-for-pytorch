#pragma once

#include <Python.h>
#include <vector>

void module_add_py_defs(std::vector<PyMethodDef>& vector, PyMethodDef* methods);
