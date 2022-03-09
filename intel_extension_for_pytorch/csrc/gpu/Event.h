#pragma once

#include <core/Event.h>
#include <torch/csrc/python_headers.h>

struct THDPEvent {
  PyObject_HEAD xpu::dpcpp::DPCPPEvent dpcpp_event;
};
extern PyObject* THDPEventClass;

void THDPEvent_init(PyObject* module);

inline bool THDPEvent_Check(PyObject* obj) {
  return THDPEventClass && PyObject_IsInstance(obj, THDPEventClass);
}
