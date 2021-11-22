#include "graph_rewrite.h"
#include "graph_rewrite_utils.h"

#include <torch/csrc/jit/frontend/code_template.h>

namespace torch {
namespace jit {
namespace graph_rewrite {

void insertPrePackedConvTranspose2dOp(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      insertPrePackedConvTranspose2dOp(block);
    }
    // TODO: add conv_transpose3d
    if (n->kind() == Symbol::fromQualString("aten::conv_transpose2d") ||
        n->kind() == Symbol::fromQualString("torch_ipex::conv_transpose2d")) {
      WithInsertPoint guard(n);
      auto graph = n->owningGraph();
      auto prepack_node = graph->create(
          Symbol::fromQualString("ipex_prepack::conv_transpose2d_prepack"), 1);
      auto input_size_option = n->inputs()
                                   .at(0)
                                   ->type()
                                   ->cast<TensorType>()
                                   ->sizes()
                                   .concrete_sizes();
      // if can't get input shape info, will not do weight prepack.
      if (!(input_size_option.has_value() &&
            input_size_option.value().size() == 4)) {
        continue;
      }
      IValue input_size_value(input_size_option.value());

      if (n->kind() == Symbol::fromQualString("aten::conv_transpose2d")) {
        auto weight_size_option = n->inputs()
                                      .at(1)
                                      ->type()
                                      ->cast<TensorType>()
                                      ->sizes()
                                      .concrete_sizes();
        // weight has not shape info, will not do weight prapacked.
        if (!(weight_size_option.has_value() &&
              weight_size_option.value().size() == 4)) {
          continue;
        }

        // TODO: output_padding unsupported in IPEX temporarily
        auto output_padding = toIValue(n->input(5))->toIntList();
        if (output_padding[0] > 0 || output_padding[1] > 0) {
          continue;
        }

        auto weight_size = weight_size_option.value();
        std::vector<int64_t> k_size = {weight_size[2], weight_size[3]};
        // w_is_channels_last is invaild, there will has a check the memory
        // format at convolution kernel side.
        bool w_is_channels_last = false;
        int64_t o_channel = weight_size[1];
        IValue kernel_size_value(k_size), weight_is_prepacked_value(false),
            weight_is_channels_last_value(w_is_channels_last),
            output_channel_value(o_channel);
        auto kernel_size = graph->insertConstant(kernel_size_value);
        auto weight_is_prepacked =
            graph->insertConstant(weight_is_prepacked_value);
        auto weight_is_channels_last =
            graph->insertConstant(weight_is_channels_last_value);
        auto output_channel = graph->insertConstant(output_channel_value);

        for (auto i = 1; i < n->inputs().size(); ++i) {
          Value* v = n->inputs().at(i);
          prepack_node->addInput(v);
        }
        prepack_node->addInput(kernel_size);
        prepack_node->addInput(output_channel);
        prepack_node->addInput(weight_is_channels_last);
        prepack_node->addInput(weight_is_prepacked);
      } else {
        for (auto i = 1; i < n->inputs().size(); ++i) {
          Value* v = n->inputs().at(i);
          prepack_node->addInput(v);
        }
      }

      auto input_size = graph->insertConstant(input_size_value);
      prepack_node->addInput(input_size);
      prepack_node->output()->setType(getCustomClass(
          "__torch__.torch.classes.ipex_prepack.ConvTransposeOpContext"));

      graph->insertNode(prepack_node);
      auto prepack_conv_transpose = graph->insertNode(graph->create(
          Symbol::fromQualString("ipex_prepack::conv_transpose2d_run"), 1));
      prepack_conv_transpose->addInput(n->inputs().at(0));
      prepack_conv_transpose->addInput(prepack_node->output());
      prepack_conv_transpose->output()->setType(
          n->output()->type()->cast<TensorType>());
      auto v = n->outputs().at(0);
      n->output()->replaceAllUsesWith(prepack_conv_transpose->output());
    }
  }
  EliminateDeadCode(b);
}

void insertPrePackedConvTranspose2dOp(std::shared_ptr<Graph>& graph) {
  insertPrePackedConvTranspose2dOp(graph->block());
}

} // namespace graph_rewrite
} // namespace jit
} // namespace torch
