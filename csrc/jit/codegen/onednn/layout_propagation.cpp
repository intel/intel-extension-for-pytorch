#include "layout_propagation.h"
#include "graph_helper.h"

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

using namespace torch::jit;

bool couldSupportOpaqueLayout(Node* node) {
  switch (node->kind()) {
    case aten::size:
      return true;
    default:
      return false;
  }
}

void LayoutPropagation(Node* n) {
  if (!LlgaGraphHelper::isLlgaSubgraph(n))
    return;

  for (auto input : n->inputs()) {
    auto prev = input->node();
    auto offset = input->offset();
    if (LlgaGraphHelper::isLlgaSubgraph(prev)) {
      bool useOpaqueLayout = true;
      for (auto& use : input->uses()) {
        if (!couldSupportOpaqueLayout(use.user) &&
            !LlgaGraphHelper::isLlgaSubgraph(use.user)) {
          useOpaqueLayout = false;
          break;
        }
      }
      if (useOpaqueLayout) {
        LlgaNodeWrapper(prev).setOpaqueLayout(offset);
      }
    }
  }
}

void LayoutPropagation(at::ArrayRef<Block*> blocks) {
  for (Block* block : blocks)
    for (Node* node : block->nodes())
      LayoutPropagation(node);
}

void PropagateLayout(const std::shared_ptr<Graph>& graph) {
  LayoutPropagation(graph->block());
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
