#pragma once

#include <core/EventBase.h>
#include <torch/csrc/python_headers.h>
#include <memory>

namespace torch_ipex::xpu {

struct THDPEvent {
  PyObject_HEAD std::shared_ptr<torch_ipex::xpu::dpcpp::DPCPPEventBase> dpcpp_event;
};
extern PyObject* THDPEventClass;

void THDPEvent_init(PyObject* module);

inline bool THDPEvent_Check(PyObject* obj) {
  return THDPEventClass && PyObject_IsInstance(obj, THDPEventClass);
}

} // namespace torch_ipex::xpu
