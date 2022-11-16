#include "Event.h"
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include "Stream.h"

namespace xpu {

PyObject* THDPEventClass = nullptr;

static PyObject* THDPEvent_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  //  unsigned char enable_timing = 0;
  //  unsigned char blocking = 0;
  //  unsigned char interprocess = 0;

  //  static char *kwlist[] =
  //    {"enable_timing", "blocking", "interprocess", nullptr};
  //  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|bbb", kwlist,
  //      &enable_timing, &blocking, &interprocess)) {
  //    return nullptr;
  //  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THDPEvent* self = (THDPEvent*)ptr.get();
  //  unsigned int flags =
  //    (blocking ? xpuEventBlockingSync : xpuEventDefault) |
  //    (enable_timing ? xpuEventDefault : xpuEventDisableTiming) |
  //    (interprocess ? xpuEventInterprocess : xpuEventDefault);

  new (&self->dpcpp_event) xpu::dpcpp::DPCPPEvent();

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THDPEvent_from_ipc_handle(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  static torch::PythonArgParser parser({
      "from_ipc_handle(Device device, std::string ipc_handle)",
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  at::Device device = r.device(0);
  std::string handle_string = r.string(1);

  //  TORCH_CHECK(handle_string.size() == sizeof(xpuIpcEventHandle_t),
  //    "xpuIpcEventHandle_t expects byte-like object of size ",
  //    sizeof(xpuIpcEventHandle_t), ", but got ", handle_string.size());
  //  TORCH_CHECK(device.type() == at::kXPU, "Event can only be created on "
  //    "XPU devices, but got device type ", device.type())

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }
  THDPEvent* self = (THDPEvent*)ptr.get();

  //  xpuIpcEventHandle_t handle;
  //  std::memcpy(&handle, handle_string.c_str(), handle_string.size());
  new (&self->dpcpp_event) xpu::dpcpp::DPCPPEvent();

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THDPEvent_dealloc(THDPEvent* self) {
  self->dpcpp_event.~DPCPPEvent();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THDPEvent_get_dpcpp_event(THDPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(nullptr);
  END_HANDLE_TH_ERRORS
}

static PyObject* THDPEvent_get_device(THDPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  at::optional<at::Device> device = self->dpcpp_event.device();
  if (!device) {
    Py_RETURN_NONE;
  }
  return THPDevice_New(device.value());
  END_HANDLE_TH_ERRORS
}

static PyObject* THDPEvent_record(THDPEvent* self, THDPStream* stream) {
  HANDLE_TH_ERRORS
  self->dpcpp_event.record(stream->dpcpp_stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THDPEvent_wait(THDPEvent* self, THDPStream* stream) {
  HANDLE_TH_ERRORS
  self->dpcpp_event.block(stream->dpcpp_stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THDPEvent_query(THDPEvent* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->dpcpp_event.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THDPEvent_elapsed_time(THDPEvent* self, THDPEvent* other) {
  HANDLE_TH_ERRORS
  return PyFloat_FromDouble(self->dpcpp_event.elapsed_time(other->dpcpp_event));
  END_HANDLE_TH_ERRORS
}

static PyObject* THDPEvent_synchronize(THDPEvent* self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    self->dpcpp_event.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THDPEvent_ipc_handle(THDPEvent* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  //  xpuIpcEventHandle_t handle;
  //  self->dpcpp_event.ipc_handle(&handle);
  //  return PyBytes_FromStringAndSize((const char *)&handle, sizeof(handle));
  return PyFloat_FromDouble(0);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef THDPEvent_properties[] = {
    {"device", (getter)THDPEvent_get_device, nullptr, nullptr, nullptr},
    {"dpcpp_event",
     (getter)THDPEvent_get_dpcpp_event,
     nullptr,
     nullptr,
     nullptr},
    {nullptr}};

static PyMethodDef THDPEvent_methods[] = {
    {(char*)"from_ipc_handle",
     (PyCFunction)(void (*)(void))THDPEvent_from_ipc_handle,
     METH_CLASS | METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {(char*)"record", (PyCFunction)THDPEvent_record, METH_O, nullptr},
    {(char*)"wait", (PyCFunction)THDPEvent_wait, METH_O, nullptr},
    {(char*)"query", (PyCFunction)THDPEvent_query, METH_NOARGS, nullptr},
    {(char*)"elapsed_time",
     (PyCFunction)THDPEvent_elapsed_time,
     METH_O,
     nullptr},
    {(char*)"synchronize",
     (PyCFunction)THDPEvent_synchronize,
     METH_NOARGS,
     nullptr},
    {(char*)"ipc_handle",
     (PyCFunction)THDPEvent_ipc_handle,
     METH_NOARGS,
     nullptr},
    {nullptr}};

PyTypeObject THDPEventType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._XPUEventBase", /* tp_name */
    sizeof(THDPEvent), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THDPEvent_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    0, /* tp_getattr */
    0, /* tp_setattr */
    0, /* tp_reserved */
    0, /* tp_repr */
    0, /* tp_as_number */
    0, /* tp_as_sequence */
    0, /* tp_as_mapping */
    0, /* tp_hash  */
    0, /* tp_call */
    0, /* tp_str */
    0, /* tp_getattro */
    0, /* tp_setattro */
    0, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    THDPEvent_methods, /* tp_methods */
    0, /* tp_members */
    THDPEvent_properties, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    0, /* tp_init */
    0, /* tp_alloc */
    THDPEvent_pynew, /* tp_new */
};

void THDPEvent_init(PyObject* module) {
  THDPEventClass = (PyObject*)&THDPEventType;
  if (PyType_Ready(&THDPEventType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THDPEventType);
  if (PyModule_AddObject(module, "_XPUEventBase", (PyObject*)&THDPEventType) <
      0) {
    throw python_error();
  }
}
} // namespace xpu
