#pragma once

#include <ATen/Tensor.h>

#include "ideep/ideep.hpp"

#include <vector>

namespace torch { namespace jit {

namespace ipex {
  static auto max_pool2d = Symbol::fromQualString("ipex::max_pool2d");
}

}} //  namespace torch::jit

namespace torch_ipex {
namespace cpu {

at::Tensor dil_max_pool2d(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode);

}  // namespace cpu
}  // namespace torch_ipex
