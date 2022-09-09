#pragma once

#include <ATen/Tensor.h>

#include <c10/core/Scalar.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

#include "ideep/ideep.hpp"

namespace torch {
namespace jit {

// XXX: PyTorch does not support nesting namespace
// And the alias analysis is not working for namespace other than aten ...
// So we fake some op namespaces to workaround that.
namespace ipex {
static auto softmax = Symbol::fromQualString("ipex::softmax");
static auto softmax_ = Symbol::fromQualString("ipex::softmax_");

} // namespace ipex

} // namespace jit
} // namespace torch

namespace torch_ipex {
namespace cpu {

at::Tensor dil_softmax(
    const at::Tensor& input,
    const int64_t dim,
    const at::IValue& dtype = at::IValue());

at::Tensor& dil_softmax_(
    at::Tensor& input,
    const int64_t dim,
    const at::IValue& dtype = at::IValue());

} // namespace cpu
} // namespace torch_ipex
