#include "layout_propagation.h"
#include <torch/csrc/jit/jit_log.h>
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

// Figure out if this use of the input is its last use.
// A node does not uniquely identify use because an input can appear
// multiple times in a node's inputs. But let's assume that the last offset of
// inputs in the list of a node's inputs would be topologically the last one in
// the subgraph
void markLastUseOfInputs(Node* n, Value* input, short inputIdx) {
  int multipleUseLastindex = -1;
  auto numUses = input->uses().size();
  for (int i = 0; i < numUses; i++) {
    auto use = input->uses()[i];
    if (((uintptr_t)n) == ((uintptr_t)(use.user))) {
      // same use as current node
      if (multipleUseLastindex == -1) {
        // the first index in the vector of this node's inputs that this input
        // is present in
        multipleUseLastindex = use.offset;
      } else if (multipleUseLastindex < use.offset) {
        // let's assume that a higher input index means that an input is used
        // later
        multipleUseLastindex = use.offset;
      }
    } else if (n->isBefore(use.user)) {
      // this node can't be the last place where this input was used
      multipleUseLastindex = -1;
      break;
    }
  }
  if (multipleUseLastindex == inputIdx) {
    // This node is the last use of this input
    // Mark last use
    auto lastUseVector = n->is(Symbol::attr("future_input_uses"));
    lastUseVector[inputIdx] = 0;
    n->is_(Symbol::attr("future_input_uses"), lastUseVector);
  }
}

void LayoutPropagation(Node* n) {
  if (!LlgaGraphHelper::isLlgaSubgraph(n))
    return;
  const auto num_inputs = n->inputs().size();
  // initial attr::output_layouts if undefined
  if (!n->hasAttribute(attr::output_layouts)) {
    const auto num_output = n->outputs().size();
    GRAPH_DEBUG("Initial output_layouts of size ", num_output);
    std::vector<int64_t> layouts(num_output, STRIDED_LAYOUT);
    n->is_(attr::output_layouts, layouts);
    // Initially, assume that graph inputs would be reused later.
    // Will assign a value of 0 if an input would not be used later.
    // and 1 no matter how many times it would be used later.
    std::vector<int64_t> future_input_uses(num_inputs, 1);
    n->is_(Symbol::attr("future_input_uses"), future_input_uses);
  }
  for (int i = 0; i < num_inputs; i++) {
    // propagate layout
    auto input = n->inputs()[i];
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
    markLastUseOfInputs(n, input, i);
  } // end outer for loop for each input
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
