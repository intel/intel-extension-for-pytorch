#pragma once

#include <core/Functions.h>


namespace xpu {
namespace dpcpp {

inline bool check_device(const at::TensorList& tensor_list) {
  if (tensor_list.empty()) {
    return true;
  }
  Device curDevice = Device(kXPU, current_device());
  for (const Tensor& t : tensor_list) {
    if (t.device() != curDevice) return false;
  }
  return true;
}

}} // namespace xpu::dpcpp
