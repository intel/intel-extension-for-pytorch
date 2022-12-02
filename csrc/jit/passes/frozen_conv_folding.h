#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch_ipex {
namespace jit {
namespace graph_rewrite {

// Fuses Convolution -> Batchnorm into a single Convolution by
// folding batchnorm weights into conv weights.
// This pass only works on Frozen Graphs; otherwise it is a No-Op.
bool FoldFrozenConvBatchnorm(std::shared_ptr<torch::jit::Graph>& graph);

// Fuses Convolution -> Add/Sub into a single Convolution by
// folding add constant tensor into conv weights.
// This pass only works on Frozen Graphs; otherwise it is a No-Op.
bool FoldFrozenConvAddOrSub(std::shared_ptr<torch::jit::Graph>& graph);

// Fuses Convolution -> Mul/Div into a single Convolution by
// folding add constant tensor into conv weights.
// This pass only works on Frozen Graphs; otherwise it is a No-Op.
bool FoldFrozenConvMulOrDiv(std::shared_ptr<torch::jit::Graph>& graph);

// Call FoldFrozenConvAddOrSub and FoldFrozenConvMulOrDiv multiple times
void FrozenConvFolding(std::shared_ptr<torch::jit::Graph>& graph);

} // namespace graph_rewrite
} // namespace jit
} // namespace torch_ipex
