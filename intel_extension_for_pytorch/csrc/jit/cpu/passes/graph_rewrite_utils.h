#pragma once

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {
namespace graph_rewrite {

bool hasSideEffectInBlocks(Block* block, Value* v);
bool hasSideEffectOrAliasInSubgraphs(Node* node, Value* v);
bool hasSideEffectOrAlias(Value* v, AliasDb* aliasDb);
MatchFilter fuse_add_filter(std::shared_ptr<Graph>& graph, int accumu_id);

} // namespace graph_rewrite
} // namespace jit
} // namespace torch
