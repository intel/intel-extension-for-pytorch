
#pragma once
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/restore_mutation.h>
#include "graph_rewrite.h"
#include "utils.h"

namespace torch {
namespace jit {
namespace graph_rewrite {
bool hasSideEffectInBlocks(Block* block, Value* v);
bool hasSideEffectOrAliasInSubgraphs(Node* node, Value* v);
bool hasSideEffectOrAlias(Value* v, AliasDb* aliasDb);
} // namespace graph_rewrite
} // namespace jit
} // namespace torch
