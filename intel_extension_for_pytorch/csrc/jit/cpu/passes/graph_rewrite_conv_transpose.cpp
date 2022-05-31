#include "graph_rewrite.h"
#include "graph_rewrite_utils.h"

#include <ATen/code_template.h>

namespace torch {
namespace jit {
namespace graph_rewrite {

using namespace at::jit;
using namespace torch_ipex::cpu;

void insertPrePackedConvTransposeOpForATen(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      insertPrePackedConvTransposeOpForATen(block);
    }
    // TODO: add conv_transpose3d
    if (n->kind() == aten::conv_transpose2d ||
        n->kind() == aten::conv_transpose3d) {
      WithInsertPoint guard(n);
      auto graph = n->owningGraph();
      auto input_size_option = n->inputs()
                                   .at(0)
                                   ->type()
                                   ->cast<TensorType>()
                                   ->sizes()
                                   .concrete_sizes();
      // if can't get input shape info, will not do weight prepack.
      if (!(input_size_option.has_value() &&
            (input_size_option.value().size() == 4 ||
             input_size_option.value().size() == 5))) {
        continue;
      }
      IValue input_size_value(input_size_option.value());

      auto weight_size_option = n->inputs()
                                    .at(1)
                                    ->type()
                                    ->cast<TensorType>()
                                    ->sizes()
                                    .concrete_sizes();
      // weight has not shape info, will not do weight prapacked.
      if (!(weight_size_option.has_value() &&
            (weight_size_option.value().size() == 4 ||
             weight_size_option.value().size() == 5))) {
        continue;
      }

      // # padding - output_padding + stride <= 0 unsupported in mkldnn
      auto stride = toIValue(n->input(3))->toIntList();
      auto padding = toIValue(n->input(4))->toIntList();
      auto output_padding = toIValue(n->input(5))->toIntList();

      auto weight_size = weight_size_option.value();
      // 2d case.
      if (weight_size.size() == 4) {
        if (padding[0] - output_padding[0] + stride[0] <= 0 ||
            padding[1] - output_padding[1] + stride[1] <= 0) {
          continue;
        }
      }
      // 3d case.
      if (weight_size.size() == 5) {
        if (padding[0] - output_padding[0] + stride[0] <= 0 ||
            padding[1] - output_padding[1] + stride[1] <= 0 ||
            padding[2] - output_padding[2] + stride[2] <= 0) {
          continue;
        }
      }

      std::vector<int64_t> k_size = {weight_size[2]};
      // 2d case.
      if (weight_size.size() == 4) {
        k_size.push_back(weight_size[3]);
      }
      // 3d case.
      if (weight_size.size() == 5) {
        k_size.push_back(weight_size[3]);
        k_size.push_back(weight_size[4]);
      }
      // w_is_channels_last is invaild, there will has a check the memory
      // format at convolution kernel side.
      bool w_is_channels_last = false;
      if (constant_as<at::Tensor>(n->namedInput("weight")).has_value()) {
        at::Tensor weight_tensor =
            constant_as<at::Tensor>(n->namedInput("weight")).value();
        w_is_channels_last =
            weight_tensor.is_contiguous(at::MemoryFormat::ChannelsLast) ||
            weight_tensor.is_contiguous(at::MemoryFormat::ChannelsLast3d);
      }
      IValue weight_is_channels_last_value(w_is_channels_last);
      auto weight_is_channels_last =
          graph->insertConstant(weight_is_channels_last_value);

      // Note that once creating this "conv_transpose_prepack" node, make sure
      // it is also inserted into the graph. Details ref to "linear_prepack"
      // creation in "graph_rewrite_linear.cpp"
      auto prepack_node = graph->create(
          Symbol::fromQualString("ipex_prepack::conv_transpose_prepack"), 1);
      for (auto i = 1; i < n->inputs().size(); ++i) {
        Value* v = n->inputs().at(i);
        prepack_node->addInput(v);
      }
      prepack_node->addInput(weight_is_channels_last);

      auto input_size = graph->insertConstant(input_size_value);
      prepack_node->addInput(input_size);
      prepack_node->output()->setType(getCustomClass(
          "__torch__.torch.classes.ipex_prepack.ConvTransposeOpContext"));

      graph->insertNode(prepack_node);
      auto prepack_conv_transpose = graph->insertNode(graph->create(
          Symbol::fromQualString("ipex_prepack::conv_transpose_run"), 1));
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

// For ipex conv_transpose, we can re-pack the packed weight in the op-context
// if we get an different input size here
void mayRePackConvTransposeOpForIpex(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      mayRePackConvTransposeOpForIpex(block);
    }
    // TODO: add conv_transpose3d
    if (n->kind() == Symbol::fromQualString("torch_ipex::conv_transpose")) {
      WithInsertPoint guard(n);
      auto graph = n->owningGraph();
      auto input_size_option = n->inputs()
                                   .at(0)
                                   ->type()
                                   ->cast<TensorType>()
                                   ->sizes()
                                   .concrete_sizes();
      // if can't get input shape info, will not do weight prepack.
      if (!(input_size_option.has_value() &&
            (input_size_option.value().size() == 4 ||
             input_size_option.value().size() == 5))) {
        continue;
      }
      IValue input_size_value(input_size_option.value());
      auto prepack_node = n->inputs().at(3)->node()->inputs().at(0);
      // For graph before "freeze", cannot get custom class to repack
      if (!toIValue(prepack_node).has_value())
        continue;
      auto convtranspose_op_ctx = toIValue(prepack_node)
                                      .value()
                                      .toCustomClass<ConvTransposeOpContext>();
      convtranspose_op_ctx->may_repack(input_size_option.value());
      auto prepack_convtranspose = graph->insertNode(graph->create(
          Symbol::fromQualString("ipex_prepack::conv_transpose_run"), 1));
      prepack_convtranspose->addInput(n->inputs().at(0));
      prepack_convtranspose->addInput(prepack_node);
      prepack_convtranspose->output()->setType(
          n->output()->type()->cast<TensorType>());
      auto v = n->outputs().at(0);
      n->output()->replaceAllUsesWith(prepack_convtranspose->output());
    }
  }
  EliminateDeadCode(b);
}

void insertPrePackedConvTransposeOp(std::shared_ptr<Graph>& graph) {
  insertPrePackedConvTransposeOpForATen(graph->block());
  mayRePackConvTransposeOpForIpex(graph->block());
}

} // namespace graph_rewrite
} // namespace jit
} // namespace torch
