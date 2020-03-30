#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {

// Base ATEN Type class where the IPE specific overrides should be defined.
class AtenIpexType {
 public:
  static void InitializeAtenBindings();
};

} // namespace torch_ipex
