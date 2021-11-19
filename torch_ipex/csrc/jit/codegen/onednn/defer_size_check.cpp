#include <torch/csrc/jit/ir/alias_analysis.h>

#include "jit/codegen/onednn/utils.h"

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

class SizeCheckMover {
 private:
  Block* block_;
  std::shared_ptr<Graph> graph_;

 public:
  SizeCheckMover(Block* block, std::shared_ptr<Graph> graph)
      : block_(block), graph_(std::move(graph)) {}

  bool analyzeNode(Node* node, AliasDb& aliasDb) {
    //
    // %b = addmm(%a)
    // %sz = aten::size(%b)
    // %c = relu(%b)
    //  =>
    // %b = addmm(%a)
    // %c = relu(%b)
    // %sz = aten::size(%c)
    //       ^-- move size check after relu as it preserves input shape
    //
    // Also support moving multiple aten::size connected to the same node
    // %b = aten::dequantize(%a)
    // %c = aten::linear(%b, %weight, %bias)
    // %sz1 = aten::size(%c, %0)
    // %sz2 = aten::size(%c, %1)
    // %d = aten::quantize_per_tensor(%c, %scale, %zp, %dtype)
    // ->
    // %b = aten::dequantize(%a)
    // %c = aten::linear(%b, %weight, %bias)
    // %d = aten::quantize_per_tensor(%c, %scale, %zp, %dtype)
    // %sz1 = aten::size(%d, %0) <--defer size after quantize_per_tensor
    // %sz2 = aten::size(%d, %1) <--defer size after quantize_per_tensor
    if (node->kind() != aten::size)
      return false;

    auto* input = node->input(0);
    auto& uses = input->uses();
    bool onlyUsedByShapePreserveOp = uses.size() > 1 &&
        std::all_of(uses.begin(), uses.end(), [node](auto& u) {
                                       return u.user == node ||
                                           u.user->kind() == aten::size ||
                                           utils::isEltwiseOp(u.user);
                                     });

    if (!onlyUsedByShapePreserveOp)
      return false;

    for (const auto& use : uses) {
      // skip the node itself and aten::size
      if (use.user == node || use.user->kind() == aten::size)
        continue;
      auto shapePreserveOp = use.user;
      if (aliasDb.moveAfterTopologicallyValid(node, shapePreserveOp)) {
        node->replaceInputWith(input, shapePreserveOp->output(0));
        return true;
      }
    }

    return false;
  }

  void run() {
    bool changed = true;
    while (changed) {
      changed = false;
      AliasDb aliasDb(graph_);
      for (Node* node : block_->nodes()) {
        changed |= analyzeNode(node, aliasDb);
      }
    }

    for (Node* node : block_->nodes())
      for (Block* subBlock : node->blocks())
        SizeCheckMover(subBlock, graph_).run();
  }
};

void DeferSizeCheck(std::shared_ptr<Graph>& graph) {
  SizeCheckMover(graph->block(), graph).run();
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
