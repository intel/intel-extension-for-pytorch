#pragma once

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

struct IPEXRemoveMutation {
  IPEXRemoveMutation(std::shared_ptr<torch::jit::Graph> graph)
      : graph_(std::move(graph)) {}

  bool removeTensorMutation();
  bool maybeAliveAfterNode(
      torch::jit::Node* node,
      torch::jit::Value* v,
      torch::jit::Value* x = nullptr);

 private:
  std::shared_ptr<torch::jit::Graph> graph_;
  std::unique_ptr<torch::jit::AliasDb> aliasDb_ = nullptr;

  torch::jit::AliasDb* getAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<torch::jit::AliasDb>(graph_);
    }
    return aliasDb_.get();
  }
  torch::jit::Node* createSpecialMappedOp(torch::jit::Node* n);
  bool removeTensorMutation(torch::jit::Block* block);
};

bool IPEXRemoveTensorMutation(const std::shared_ptr<torch::jit::Graph>& graph);

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
