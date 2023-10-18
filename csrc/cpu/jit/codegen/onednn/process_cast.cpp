#include "process_cast.h"
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

using namespace torch::jit;

// oneDNN Graph does not support TypeCast with the same output dtype as the
// input dtype.
static void RemoveRedundantCast(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      RemoveRedundantCast(sub);
    }

    if (((node->kind() == aten::to) || (node->kind() == aten::type_as)) &&
        node->input(0)->type()->isSubtypeOf(TensorType::get()) &&
        node->input(0)->type()->cast<TensorType>()->scalarType().has_value()) {
      auto dtypeOfInput =
          node->input(0)->type()->cast<TensorType>()->scalarType().value();
      auto dtypeOfOutput =
          node->output(0)->type()->cast<TensorType>()->scalarType();
      if (dtypeOfOutput.has_value()) {
        if (dtypeOfInput == dtypeOfOutput.value())
          node->output()->replaceAllUsesWith(node->input(0));
      }
    }
  }
}

// oneDNN Graph expects the input dtype of softmax to be same as its output type
static void InsertCastForInconsistentInputOutput(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      InsertCastForInconsistentInputOutput(sub);
    }

    if (node->kind() == aten::softmax &&
        node->input(0)->type()->isSubtypeOf(TensorType::get()) &&
        node->input(0)->type()->cast<TensorType>()->scalarType().has_value()) {
      auto dtypeOfInput =
          node->input(0)->type()->cast<TensorType>()->scalarType().value();
      auto dtypeOfOutput =
          node->output(0)->type()->cast<TensorType>()->scalarType();
      if (dtypeOfOutput.has_value()) {
        if (dtypeOfInput != dtypeOfOutput.value()) {
          WithInsertPoint guard(node);
          auto g = node->owningGraph();
          auto cast =
              g->insert(aten::to, {node->input(0), dtypeOfOutput.value()});
          cast->setType(node->output(0)->type());
          node->replaceInput(0, cast);
        }
      }
    }
  }
}

void ProcessCast(std::shared_ptr<torch::jit::Graph>& graph) {
  InsertCastForInconsistentInputOutput(graph->block());
  RemoveRedundantCast(graph->block());
  EliminateDeadCode(graph);
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
