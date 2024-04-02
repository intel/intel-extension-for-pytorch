#pragma once

#include <Macros.h>
#include <torch/csrc/jit/ir/ir.h>
#include "graph_rewrite.h"
#include "graph_rewrite_utils.h"

namespace torch_ipex {
namespace jit {
namespace graph_rewrite {

struct IPEX_API LinearBNParameters {
  at::Tensor linear_w;
  at::Tensor linear_b;
  at::Tensor bn_rm;
  at::Tensor bn_rv;
  double bn_eps = 0.0;
  at::Tensor bn_w;
  at::Tensor bn_b;
};

/**
 * Given the current weight and bias tensors of a Linear module and parameters
 * of the BatchNorm module we're folding with, compute the updated values
 * for the weight and bias.
 *
 * The function is basically copied from torch/nn/utils/fusion.py
 */
IPEX_API std::tuple<at::Tensor, at::Tensor> computeUpdatedLinearWeightAndBias(
    const LinearBNParameters& p);

// Fuses Linear -> BatchNormNd into a single Linear by
// folding batchnorm weights into linear weights.
// This pass only works on Frozen Graphs; otherwise it is a No-Op.
bool FoldFrozenLinearBatchnorm(std::shared_ptr<torch::jit::Graph>& graph);

// Fuses Linear -> Add/Sub into a single Linear by
// folding add constant tensor into linear weights.
// This pass only works on Frozen Graphs; otherwise it is a No-Op.
bool FoldFrozenLinearAddOrSub(std::shared_ptr<torch::jit::Graph>& graph);

// Fuses Linear -> Mul/Div into a single Linear by
// folding add constant tensor into linear weights.
// This pass only works on Frozen Graphs; otherwise it is a No-Op.
bool FoldFrozenLinearMulOrDiv(std::shared_ptr<torch::jit::Graph>& graph);

// Call FoldFrozenLinearAddOrSub and FoldFrozenLinearMulOrDiv multiple times
void FrozenLinearFolding(std::shared_ptr<torch::jit::Graph>& graph);

} // namespace graph_rewrite
} // namespace jit
} // namespace torch_ipex
