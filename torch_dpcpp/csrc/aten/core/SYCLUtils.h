#pragma once

#include <core/SYCLContext.h>


namespace at { namespace sycl {

// Check if every tensor in a list of tensors matches the current
// device.
inline bool check_device(ArrayRef<Tensor> ts) {
  if (ts.empty()) {
    return true;
  }
  Device curDevice = Device(kSYCL, current_device());
  for (const Tensor& t : ts) {
    if (t.device() != curDevice) return false;
  }
  return true;
}

}} //namespace at::sycl
