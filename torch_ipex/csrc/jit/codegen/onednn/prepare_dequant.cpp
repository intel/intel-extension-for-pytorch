#include "jit/codegen/onednn/prepare_dequant.h"

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

void mayDecomposeDequant(Node* node) {
  if (node->input(0)->node()->kind()!= Symbol::aten("quantize_per_tensor") && node->input(0)->node()->kind()!= Symbol::aten("quantize_per_channel")) {
      return;
  }

  auto* output = node->output(0);
  auto& uses = output->uses();
  if (uses.size() < 2) {
      return;
  }

  WithInsertPoint guard(node);
  auto g = node->owningGraph();
  int nb_uses = uses.size();

  for (int i = 1; i < nb_uses; i++) {
    auto dequant = g->insert(Symbol::aten("dequantize"), {node->input(0)});
    auto right_fork =  uses[i].user;
    right_fork->replaceInputWith(output, dequant);
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
