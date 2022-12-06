#pragma once

#include <core/EventBase.h>
#include <torch/csrc/python_headers.h>
#include <memory>

namespace xpu {

struct THDPEvent {
  PyObject_HEAD std::shared_ptr<xpu::dpcpp::DPCPPEventBase> dpcpp_event;
};
extern PyObject* THDPEventClass;

void THDPEvent_init(PyObject* module);

inline bool THDPEvent_Check(PyObject* obj) {
  return THDPEventClass && PyObject_IsInstance(obj, THDPEventClass);
}
} // namespace xpu
