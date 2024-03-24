#pragma once

#include <memory>

#include <ATen/TensorUtils.h>
#include <c10/core/Device.h>
#include <c10/xpu/XPUFunctions.h>
#include <runtime/Device.h>
#include <utils/Macros.h>

using namespace at;

namespace torch_ipex::xpu {
namespace dpcpp {

static inline void isOnSameDevice(
    at::CheckedFrom c,
    const at::TensorArg& t1,
    const at::TensorArg& t2) {
  if ((t1->device().type() != at::kXPU) || (t2->device().type() != at::kXPU)) {
    std::ostringstream oss;
    if (t1->device().type() != at::kXPU) {
      oss << "Tensor for " << t1 << " is not on XPU, ";
    }
    if (t2->device().type() != at::kXPU) {
      oss << "Tensor for " << t2 << " is not on XPU, ";
    }
    oss << "but expected "
        << ((!(t1->device().type() == at::kXPU ||
               t2->device().type() == at::kXPU))
                ? "them"
                : "it")
        << " to be on XPU (while checking arguments for " << c << ")";
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

static inline bool isOnSameDevice(const at::TensorList& tensor_list) {
  if (tensor_list.empty()) {
    return true;
  }
  Device curDevice = Device(kXPU, at::xpu::current_device());
  for (const Tensor& t : tensor_list) {
    if (t.device() != curDevice)
      return false;
  }
  return true;
}

} // namespace dpcpp
} // namespace torch_ipex::xpu
