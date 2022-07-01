#include <ATen/code_template.h>
#include "graph_rewrite.h"

namespace torch_ipex {
namespace jit {
namespace graph_rewrite {

using namespace at::jit;
using namespace torch::jit;

// This code will be removed after the official PyTorch NNC fully support
// BFloat16.

void replaceAtenToWithIPEXTo(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      replaceAtenToWithIPEXTo(block);
    }
    if (n->kind() == aten::to) {
      // skip aten::to.other
      if (n->inputs().at(1)->type()->kind() == TypeKind::TensorType) {
        continue;
      }
      if (n->inputs().size() == 5 || n->inputs().size() == 4) {
        auto const& input_dtype =
            n->inputs().at(0)->type()->cast<TensorType>()->scalarType();
        auto const& output_dtype =
            n->outputs().at(0)->type()->cast<TensorType>()->scalarType();
        if (!input_dtype || !output_dtype) {
          continue;
        }
        if (!(*input_dtype == c10::ScalarType::Float &&
              *output_dtype == c10::ScalarType::BFloat16)) {
          continue;
        }
        // device check?
        WithInsertPoint guard(n);
        auto graph = n->owningGraph();
        Node* ipex_to_node =
            graph->create(Symbol::fromQualString("ipex::to_dtype"));
        for (auto i = 0; i < n->inputs().size(); ++i) {
          Value* v = n->inputs().at(i);
          ipex_to_node->addInput(v);
        }
        graph->insertNode(ipex_to_node);
        n->output()->replaceAllUsesWith(ipex_to_node->output());
      } else {
        continue;
      }
    }
  }
  EliminateDeadCode(b);
}

void replaceIPEXToWithAtenTo(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      replaceIPEXToWithAtenTo(block);
    }
    if (n->kind() == Symbol::fromQualString("ipex::to_dtype")) {
      WithInsertPoint guard(n);
      auto graph = n->owningGraph();
      Node* aten_to_node = graph->create(aten::to);
      for (auto i = 0; i < n->inputs().size(); ++i) {
        Value* v = n->inputs().at(i);
        aten_to_node->addInput(v);
      }
      graph->insertNode(aten_to_node);
      n->output()->replaceAllUsesWith(aten_to_node->output());
    }
  }
  EliminateDeadCode(b);
}

void replaceAtenToWithIPEXTo(std::shared_ptr<Graph>& graph) {
  replaceAtenToWithIPEXTo(graph->block());
  EliminateDeadCode(graph);
}

void replaceIPEXToWithAtenTo(std::shared_ptr<Graph>& graph) {
  replaceIPEXToWithAtenTo(graph->block());
  EliminateDeadCode(graph);
}

} // namespace graph_rewrite
} // namespace jit
} // namespace torch_ipex
