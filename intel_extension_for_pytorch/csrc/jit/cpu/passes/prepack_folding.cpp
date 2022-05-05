#include <torch/csrc/jit/passes/constant_propagation.h>
#include "csrc/jit/cpu/kernels/OpContext.h"

#include "prepack_folding.h"

namespace torch {
namespace jit {

void PrePackingOpsFolder(Block* b) {
  auto is_foldable_op = [](const Node* n) -> bool {
    return (
        n->kind() ==
            Symbol::fromQualString("ipex_prepack::convolution_prepack") ||
        n->kind() ==
            Symbol::fromQualString("ipex_prepack::convolution_relu_prepack") ||
        n->kind() ==
            Symbol::fromQualString(
                "ipex_prepack::convolution_sigmoid_prepack") ||
        n->kind() ==
            Symbol::fromQualString("ipex_prepack::convolution_swish_prepack") ||
        n->kind() ==
            Symbol::fromQualString("ipex_prepack::convolution_elu_prepack") ||
        n->kind() ==
            Symbol::fromQualString(
                "ipex_prepack::convolution_hardtanh_prepack") ||
        n->kind() ==
            Symbol::fromQualString("ipex_prepack::convolution_add_prepack") ||
        n->kind() ==
            Symbol::fromQualString(
                "ipex_prepack::convolution_add_relu_prepack") ||

        n->kind() == Symbol::fromQualString("ipex_prepack::linear_prepack") ||
        n->kind() ==
            Symbol::fromQualString("ipex_prepack::conv_transpose_prepack") ||
        n->kind() ==
            Symbol::fromQualString(
                "ipex_prepack::convolution_leaky_relu_prepack") ||
        n->kind() ==
            Symbol::fromQualString("ipex_prepack::convolution_gelu_prepack"));
  };

  std::unordered_set<Node*> nodes_to_delete;
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      PrePackingOpsFolder(block);
    }
    if (is_foldable_op(n)) {
      auto optional_outputs = torch::jit::runNodeIfInputsAreConstant(n);
      if (optional_outputs) {
        auto outputs = optional_outputs.value();
        TORCH_CHECK(outputs.size() == 1, "Prepack ops have single output");
        Value* prepack_op_value = n->output(0);
        auto graph = n->owningGraph();
        WithInsertPoint ins(prepack_op_value->node());
        // make sure objects inserted into the graph do not holding owning
        // reference, see more details in
        // https://github.com/pytorch/pytorch/pull/65442, so there we convert
        // the object to to weak_compilation.
        auto weak_class_obj =
            outputs[0].toObject()->copy_to_weak_compilation_ref();
        Value* packed_weight = graph->insertConstant(weak_class_obj)
                                   ->setType(n->output(0)->type());
        prepack_op_value->replaceAllUsesWith(packed_weight);
        nodes_to_delete.insert(n);
      }
    }
  }
  for (auto n : nodes_to_delete) {
    n->removeAllInputs();
  }
  for (auto n : nodes_to_delete) {
    n->destroy();
  }
}

void PrePackingOpsFolder(std::shared_ptr<Graph>& graph) {
  PrePackingOpsFolder(graph->block());
}

} // namespace jit
} // namespace torch
