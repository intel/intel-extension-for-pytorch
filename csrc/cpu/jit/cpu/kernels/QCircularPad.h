
#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {
namespace cpu {

at::Tensor qpad_circular(const at::Tensor& self, c10::IntArrayRef padding);

} // namespace cpu
} // namespace torch_ipex
