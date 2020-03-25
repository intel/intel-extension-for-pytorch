#pragma once

#include <ATen/TensorUtils.h>
#include <c10/core/DeviceType.h>

static inline void checkSameDPCPP(
    at::CheckedFrom c,
    const at::TensorArg& t1,
    const at::TensorArg& t2) {
  if ((t1->device().type() != at::kDPCPP) ||
      (t2->device().type() != at::kDPCPP)) {
    std::ostringstream oss;
    if (t1->device().type() != at::kDPCPP) {
      oss << "Tensor for " << t1 << " is not on DPCPP, ";
    }
    if (t2->device().type() != at::kDPCPP) {
      oss << "Tensor for " << t2 << " is not on DPCPP, ";
    }
    oss << "but expected " << ((!(t1->device().type() == at::kDPCPP ||
                                  t2->device().type() == at::kDPCPP))
                                   ? "them"
                                   : "it")
        << " to be on DPCPP (while checking arguments for " << c << ")";
    TORCH_CHECK(0, oss.str());
  }
  TORCH_CHECK(
      t1->get_device() == t2->get_device(),
      "Expected tensor for ",
      t1,
      " to have the same device as tensor for ",
      t2,
      "; but device ",
      t1->get_device(),
      " does not equal ",
      t2->get_device(),
      " (while checking arguments for ",
      c,
      ")");
}
