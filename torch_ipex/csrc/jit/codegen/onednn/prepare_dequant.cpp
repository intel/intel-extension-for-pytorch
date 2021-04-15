#include "jit/codegen/onednn/prepare_dequant.h"

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

void mayDecomposeDequant(Node* node) {
  if (node->input(0)->node()->kind() != Symbol::aten("quantize_per_tensor") && node->input(0)->node()->kind() != Symbol::aten("quantize_per_channel")) {
      return;
  }

  auto* quant_node = node->input(0);

  auto* dequant = node->output(0);
  auto& uses = dequant->uses();
  if (uses.size() < 2) {
      return;
  }

  WithInsertPoint guard(node);
  auto g = node->owningGraph();
  int nb_uses = uses.size();

  // save the dequant_users before modifying the graph
  std::vector<torch::jit::Node*> dequant_users;
  for (const auto& use : uses) {
    dequant_users.push_back(use.user);
  }

  for (int i = 1; i < nb_uses; i++) {
    auto split_dequant = g->insert(Symbol::aten("dequantize"), {quant_node});
    auto dequant_user =  dequant_users[i];
    dequant_user->replaceInputWith(dequant, split_dequant);
  }
}

static void DecomposeDequant(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      DecomposeDequant(sub);
    }

    if (node->kind() == Symbol::aten("dequantize")) {
      mayDecomposeDequant(node);
    }
  }
}

void PrepareDequantForLLGA(std::shared_ptr<Graph>& graph) {
  DecomposeDequant(graph->block());
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
