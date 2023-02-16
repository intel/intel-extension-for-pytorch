#include "operation_syclkernel.h"
#include <torch/extension.h>
#include <iostream>

TinyTensor run_syclkernel_operation_scaledown(
    const TinyTensor& inp,
    sycl::queue* q) {
  TinyTensor outp(inp.N, inp.C, inp.H / 2, inp.W / 2);

  float* src = inp.data;
  float* dst = outp.data;

  q->submit([&](sycl::handler& h) {
    h.parallel_for(outp.count(), [=](sycl::item<1> item) {
      int idx = item.get_id(0);
      dst[idx] = src[idx * 4];
    });
  });

  return outp;
}

PYBIND11_MODULE(operation_syclkernel, m) {
  m.def(
      "run_syclkernel_operation_scaledown",
      &run_syclkernel_operation_scaledown,
      "run_syclkernel_operation_scaledown");
}
