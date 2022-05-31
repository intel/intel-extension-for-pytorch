#include "remove_redundant_aliases.h"
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

namespace torch {
namespace jit {

void RemoveRedundantAliases(const std::shared_ptr<Graph>& graph) {
  // DBR quantization uses torch.Tensor.as_subclass frequently. When
  // the quantized model is traced with torch.jit.trace, these calls appear
  // in the resulting graph as aten::alias. This PR adds a pass to remove
  // these calls from the graph.
  // Replace
  //
  // %b = aten::alias(%a)
  // %c = foo(%b)
  //
  // with
  //
  // %c = foo(%a)

  auto g = graph;
  const bool is_frozen = false;
  const bool descend_function_calls = true;
  AliasDb alias_db(g, is_frozen, descend_function_calls);
  // find the alias nodes
  std::vector<Node*> alias_nodes;
  DepthFirstGraphNodeIterator it(g);
  Node* node = nullptr;
  while ((node = it.next()) != nullptr) {
    if (node->kind() == Symbol::aten("alias")) {
      alias_nodes.push_back(node);
    }
  }

  // remove the alias nodes
  for (auto* node : alias_nodes) {
    GRAPH_DEBUG(*node);

    Value* input_value = node->input();
    Value* output_value = node->output();
    // output and input always share same memory,
    // is it safe to direct remove the node?
    output_value->replaceAllUsesWith(input_value);
    node->destroy();
  }
}

} // namespace jit
} // namespace torch
