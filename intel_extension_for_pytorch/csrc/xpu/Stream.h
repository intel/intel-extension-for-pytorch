#pragma once

#include <core/Stream.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>

namespace xpu {

struct THDPStream {
  struct THPStream base;
  xpu::dpcpp::DPCPPStream dpcpp_stream;
};
extern PyObject* THDPStreamClass;

void THDPStream_init(PyObject* module);

inline bool THDPStream_Check(PyObject* obj) {
  return THDPStreamClass && PyObject_IsInstance(obj, THDPStreamClass);
}

} // namespace xpu
