//
// Created by johnlu on 2020/11/6.
//
#include "py_init.h"

PYBIND11_MODULE(_C, m) {
  torch_ipex_init(m);
}