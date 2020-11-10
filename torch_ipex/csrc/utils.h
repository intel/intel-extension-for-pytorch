#ifndef THP_UTILS_H
#define THP_UTILS_H

#include <Python.h>
#include <vector>
void module_add_py_defs(std::vector<PyMethodDef>& vector, PyMethodDef* methods);

#endif
