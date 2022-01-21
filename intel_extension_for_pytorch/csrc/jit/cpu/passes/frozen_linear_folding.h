#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Fuses Linear -> Add/Sub into a single Linear by
// folding add constant tensor into linear weights.
// This pass only works on Frozen Graphs; otherwise it is a No-Op.
TORCH_API void FoldFrozenLinearAddOrSub(std::shared_ptr<Graph>& graph);

// Fuses Linear -> Mul/Div into a single Linear by
// folding add constant tensor into linear weights.
// This pass only works on Frozen Graphs; otherwise it is a No-Op.
TORCH_API void FoldFrozenLinearMulOrDiv(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
