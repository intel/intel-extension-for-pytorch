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
static auto einsum_binary = Symbol::fromQualString("ipex::einsum_binary");

} // namespace ipex

} // namespace jit
} // namespace torch

namespace torch_ipex {
namespace cpu {

at::Tensor einsum_binary(
    c10::string_view,
    const c10::List<at::Tensor>& operands,
    const at::Tensor& input,
    const c10::Scalar& alpha);

bool is_add_broadcast_supported_by_onednn(
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& post_add_tensor);
} // namespace cpu
} // namespace torch_ipex
