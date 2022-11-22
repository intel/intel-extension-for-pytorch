#include "prepare_silu.h"
#include "operator.h"

#include <ATen/code_template.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

using namespace torch::jit;

bool shouldDecomposeSilu(Node* node) {
  if (node->kind() != aten::silu) {
    return false;
  }
  auto inputToSilu = node->input(0)->node()->kind();
  if ((inputToSilu == aten::_convolution) || (inputToSilu == aten::linear)) {
    return true;
  }
  return false;
}

void DecomposeSilu(Node* node) {
  if (shouldDecomposeSilu(node)) {
    auto dtype = node->input(0)->type()->expect<TensorType>();

    WithInsertPoint guard(node);
    auto g = node->owningGraph();
    auto sigmoid = g->insert(aten::sigmoid, {node->input(0)});
    sigmoid->setType(dtype);

    auto mul = g->insert(aten::mul, {sigmoid, node->input(0)});
    mul->setType(dtype);

    node->output()->replaceAllUsesWith(mul);
  }
}

static void DecomposeSilu(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      DecomposeSilu(sub);
    }

    if (node->kind() == aten::silu) {
      DecomposeSilu(node);
    }
  }
}

void PrepareSiluForLLGA(std::shared_ptr<Graph>& graph) {
  DecomposeSilu(graph->block());
  EliminateDeadCode(graph);
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
