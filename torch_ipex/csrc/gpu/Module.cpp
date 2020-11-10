//
// Created by johnlu on 2020/10/21.
//
#include <Module.h>
#include <torch/csrc/THP.h>
#include <Stream.h>
#include <torch/csrc/Exceptions.h>
#include <core/Functions.h>

PyObject* module;

PyObject * THPModule_setDevice_wrap(PyObject *self, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to setDevice");
  int64_t device = THPUtils_unpackLong(arg);

  at::dpcpp::set_device(static_cast<c10::DeviceIndex>(device));

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPModule_getDevice_wrap(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto device = static_cast<int>(at::dpcpp::current_device());
  return PyLong_FromLong(device);
  END_HANDLE_TH_ERRORS
}

PyObject * THPModule_getDeviceCount_wrap(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  return PyLong_FromLong(at::dpcpp::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject * THPModule_getCurrentStream_wrap(
  PyObject * /* unused */, PyObject *device_index) {
  HANDLE_TH_ERRORS
      THPUtils_assert(
        THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
      int64_t device = THPUtils_unpackLong(device_index);
      return PyLong_FromUnsignedLongLong(
        at::dpcpp::getCurrentDPCPPStream(device).pack());
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef _THCPModule_methods[] = {
//  {"_cuda_init",        (PyCFunction)THCPModule_initExtension,    METH_NOARGS,  nullptr},
  {"_setDevice",   (PyCFunction)THPModule_setDevice_wrap,   METH_O,       nullptr},
  {"_getDevice",   (PyCFunction)THPModule_getDevice_wrap,   METH_NOARGS,  nullptr},
  {"_getDeviceCount", (PyCFunction)THPModule_getDeviceCount_wrap, METH_NOARGS, nullptr},
  {"_getCurrentStream", (PyCFunction)THPModule_getCurrentStream_wrap, METH_O, nullptr},
  {nullptr}
};

void init_module(pybind11::module& m) {
//  py::object torch = py::module::import("torch");
//  py::object torch_C = torch.attr("_C");
  pybind11::module torch_ipex_c = m.def_submodule("_C");

  THDPStream_init(torch_ipex_c.ptr());
  PyModule_AddFunctions(torch_ipex_c.ptr(), _THCPModule_methods);
}