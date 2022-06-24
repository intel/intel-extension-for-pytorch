#pragma once

#include <ATen/ATen.h>

namespace torch_ipex {
namespace cpu {

inline at::Tensor dil_hardsigmoid(const at::Tensor& self) {
  return at::hardsigmoid(self);
}

} // namespace cpu
} // namespace torch_ipex
