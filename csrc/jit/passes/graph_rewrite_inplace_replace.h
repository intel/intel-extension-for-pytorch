
#pragma once
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/restore_mutation.h>
#include "graph_rewrite.h"
#include "utils.h"

namespace torch_ipex {
namespace jit {
namespace graph_rewrite {

bool hasSideEffectInBlocks(torch::jit::Block* block, torch::jit::Value* v);
bool hasSideEffectOrAliasInSubgraphs(
    torch::jit::Node* node,
    torch::jit::Value* v);
bool hasSideEffectOrAlias(torch::jit::Value* v, torch::jit::AliasDb* aliasDb);

} // namespace graph_rewrite
} // namespace jit
} // namespace torch_ipex
