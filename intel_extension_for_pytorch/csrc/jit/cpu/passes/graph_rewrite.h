#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {
namespace graph_rewrite {

// those code just copy from PyTorch offical:
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/passes/graph_rewrite_helper.h

Value* getValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap);
c10::optional<IValue> getIValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap);

void FuseShuffle(std::shared_ptr<Graph>& graph);
void FuseMHAScoreCalc(std::shared_ptr<Graph>& graph);
void replaceAtenMaxPool2dWithIpexMaxPool2d(std::shared_ptr<Graph>& graph);
void replaceOpsWithAtenInplaceOps(std::shared_ptr<Graph>& graph);
void replaceAtenOpsWithIpexInplaceOps(std::shared_ptr<Graph>& graph);
void replaceAtenSoftmaxWithIpexSoftmax(std::shared_ptr<Graph>& graph);
void replaceAtenBatchNormWithIpexBatchNorm(std::shared_ptr<Graph>& graph);
void replaceAtenLayerNormWithIpexLayerNorm(std::shared_ptr<Graph>& graph);
void replaceEmbeddingBagWithQEmbeddingBag(std::shared_ptr<Graph>& graph);
void replaceInteractionWithQInteraction(std::shared_ptr<Graph>& graph);

void insertPrePackedConvOp(std::shared_ptr<Graph>& graph);
void fuseConvWithEltwise(std::shared_ptr<Graph>& graph);
void fuseConvAddRelu(std::shared_ptr<Graph>& graph);
void fuseBottleneck(std::shared_ptr<Graph>& graph);

void insertPrePackedLinearOp(std::shared_ptr<Graph>& graph);
void fuseLinearWithEltwise(std::shared_ptr<Graph>& graph);
void fuseLinearAddRelu(std::shared_ptr<Graph>& graph);

void FuseAddLayerNorm(std::shared_ptr<Graph>& graph);
void FuseConcatBnRelu(std::shared_ptr<Graph>& graph);

void insertPrePackedConvTranspose2dOp(std::shared_ptr<Graph>& graph);

} // namespace graph_rewrite
} // namespace jit
} // namespace torch
