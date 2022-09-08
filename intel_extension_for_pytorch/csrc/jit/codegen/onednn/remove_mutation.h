#pragma once

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch_ipex {
namespace jit {
/** This function tries to check if the mutated value v except its alias x is
 * still alive after node.
 *
 * @param aliasdb: The aliasdb of the graph owned by node
 * @param node: The node of an inplace op
 * @param v: The first input of the node
 * @param x: An alias of v, its use is excluded from the check
 *
 **/
bool maybeAliveAfterNode(
    torch::jit::AliasDb* aliasdb,
    torch::jit::Node* node,
    torch::jit::Value* v,
    torch::jit::Value* x = nullptr);
namespace fuser {
namespace onednn {

struct IPEXRemoveMutation {
  IPEXRemoveMutation(std::shared_ptr<torch::jit::Graph> graph)
      : graph_(std::move(graph)) {}

  bool removeTensorMutation();

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
