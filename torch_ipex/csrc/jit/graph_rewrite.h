#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>

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
void replaceConvolutionWithAtenConv(std::shared_ptr<Graph>& graph);
void replaceAtenConvolutionWithIpexConv(std::shared_ptr<Graph>& graph);
void FuseConvolutionWithEltwise(std::shared_ptr<Graph>& graph);
void FuseShuffle(std::shared_ptr<Graph>& graph);
void replaceAtenMaxPool2dWithIpexMaxPool2d(std::shared_ptr<Graph>& graph);
void replaceAtenLinearWithIpexLinear(std::shared_ptr<Graph>& graph);

} // namespace graph_rewrite_helper
} // namespace jit
} // namespace torch

