#include <ATen/Utils.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>

#include "qpadding.h"

namespace torch_ipex {
namespace jit {

using namespace torch::jit;

void QPaddingConversion(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      QPaddingConversion(block);
    }
    // convert q->dq->aten::pad->aten::_convolution to
    // q->dq->aten::pad->q-dq->aten::_convolution.
    if (n->kind() == aten::pad) {
      if (n->output()->uses().size() > 1 ||
          n->output()->uses().at(0).user->kind() != aten::_convolution ||
          n->inputs().at(0)->node()->kind() != aten::dequantize) {
        continue;
      }
      // make sure has type info.
      if (n->output()->type()->cast<TensorType>() == nullptr ||
          n->output()->type()->cast<TensorType>()->scalarType() ==
              c10::nullopt) {
        continue;
      }
      WithInsertPoint guard(n);
      auto conv_node = n->output()->uses().at(0).user;
      auto quantize_node = n->inputs().at(0)->node()->inputs().at(0)->node();
      auto quantize_type = quantize_node->output()
                               ->type()
                               ->cast<c10::TensorType>()
                               ->scalarType();
      auto g = n->owningGraph();

      auto new_quantize_node =
          g->create(aten::quantize_per_tensor, 1)->insertAfter(n);
      new_quantize_node->addInput(n->output());
      new_quantize_node->addInput(quantize_node->inputs().at(1));
      new_quantize_node->addInput(quantize_node->inputs().at(2));
      new_quantize_node->addInput(quantize_node->inputs().at(3));
      new_quantize_node->output()->setType(
          n->output()->type()->expect<TensorType>()->withScalarType(
              quantize_type));

      auto new_dequantize_node =
          g->create(aten::dequantize, 1)->insertAfter(new_quantize_node);
      new_dequantize_node->addInput(new_quantize_node->output());
      new_dequantize_node->output()->setType(n->output()->type());

      conv_node->replaceInputWith(n->output(), new_dequantize_node->output());
    }
  }
}

void QPaddingConversion(std::shared_ptr<torch::jit::Graph>& graph) {
  QPaddingConversion(graph->block());
}

} // namespace jit
} // namespace torch_ipex
