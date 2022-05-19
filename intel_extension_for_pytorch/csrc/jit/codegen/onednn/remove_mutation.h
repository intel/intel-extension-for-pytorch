#pragma once

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

struct IPEXRemoveMutation {
  IPEXRemoveMutation(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

  bool removeTensorMutation();
  bool maybeAliveAfterNode(Node* node, Value* v, Value* x = nullptr);

 private:
  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;

  AliasDb* getAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }
  Node* createSpecialMappedOp(Node* n);
  bool removeTensorMutation(Block* block);
};

bool IPEXRemoveTensorMutation(const std::shared_ptr<Graph>& graph);

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch