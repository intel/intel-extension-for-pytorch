#pragma once

#include <ATen/Tensor.h>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

at::Tensor pooling_impl(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    ideep::algorithm algo);

} // namespace cpu
} // namespace torch_ipex
