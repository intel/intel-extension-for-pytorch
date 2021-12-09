#pragma once

#include <ATen/Tensor.h>

#include <c10/core/Scalar.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

#include "csrc/cpu/ideep/ideep.hpp"

namespace torch {
namespace jit {

// XXX: PyTorch does not support nesting namespace
// And the alias analysis is not working for namespace other than aten ...
// So we fake some op namespaces to workaround that.
namespace ipex {
static auto max_pool2d = Symbol::fromQualString("ipex::max_pool2d");

} // namespace ipex

} // namespace jit
} // namespace torch

namespace torch_ipex {
namespace cpu {

// for JIT ops
at::Tensor dil_max_pool2d(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode);

} // namespace cpu
} // namespace torch_ipex
