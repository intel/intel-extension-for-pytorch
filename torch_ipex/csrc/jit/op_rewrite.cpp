#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/passes/constant_propagation.h>

#include <ideep.hpp>

#include "graph_ext.h"
#include "op_rewrite.h"
#include "accelerated_ops.h"

namespace torch { namespace jit {

NodeExt* replaceOpWithDNNL(Node *node, Graph *g) {
  static const std::unordered_map<NodeKind, NodeKind> rules = {
    { aten::conv2d, dnnl::conv2d },
    { aten::relu, dnnl::relu },
    { Symbol::fromQualString("aten::relu_"), dnnl::relu_ },
    { aten::batch_norm, dnnl::batch_norm },
    { aten::max_pool2d, dnnl::pooling_max_2d },
    { aten::avg_pool2d, dnnl::pooling_avg_2d },
    { aten::add, dnnl::sum },
    { aten::add_, dnnl::sum_ }
  };

  auto* replacement = reinterpret_cast<NodeExt *>(
      replaceOpWithNewKind(node, g, rules.at(node->kind())));
  replacement->initFormatInfo();
  return replacement;
}

void OpRewritePass(Block *block) {
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto *node = *it;
    ++it;

    for (auto sub : it->blocks()) {
      OpRewritePass(sub);
    }

    // Match convolution
    if (node ->matches(
          "aten::conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor")) {
      //
      // Convert it to dnnl::conv2d and insert reorder at the end of
      // it. Because dnnl::conv2d will generate opaque tensor and we
      // need a reorder to transform it back
      //
      auto newNode = replaceOpWithDNNL(node, block->owningGraph());
      auto conv2d = newNode->cast<Conv2dNode>();
      conv2d->fixWeightFormatIfPossible();
      conv2d->appendReorder(natureFormat);

      // If we could get more information about the weights
      // We could prepend a reorder for the weights and constant propagation
      // might help us create a MKL-DNN friendly weight
    } else if (node->matches("aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
        || node->matches("aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)")) {
      //
      // Add is a versatile operator so we should carefully consider whether
      // to substitue it or not
      //
      auto lh_node = node->input(0)->node();
      auto rh_node = node->input(1)->node();
      auto by_pass_reorder = [](const Node *n) {
        return (n->kind() == dnnl::reorder)
          ? n->input()->node() : n;
      };

      //
      // higher priority for conv+sum fusion than other kind
      // possibly we check whether there is a chance for conv+sum+relu
      //
      if (by_pass_reorder(lh_node)->kind() == dnnl::conv2d
          || by_pass_reorder(rh_node)->kind() == dnnl::conv2d
          || by_pass_reorder(lh_node)->kind() == dnnl::batch_norm
          || by_pass_reorder(rh_node)->kind() == dnnl::batch_norm)
        replaceOpWithDNNL(node, block->owningGraph());
    } else if (node->matches("aten::relu(Tensor self) -> Tensor")
        || node->matches("aten::relu_(Tensor(a!) self) -> Tensor(a!)")
        || node->matches(
          "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor")) {
      //
      // These ops accept both CPU tensor and opaque tensor and act accordingly
      // So we don't need to prepend or append reorders
      //
      replaceOpWithDNNL(node, block->owningGraph());
    } else if (node->matches("aten::to_dense(Tensor self) -> Tensor")
        || node->matches("aten::to_mkldnn(Tensor self) -> Tensor")) {
      //
      // Nullity both ops at current stage
      //
      node->output()->replaceAllUsesWith(node->input());

    } else if (node->matches("aten::max_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor")) {
      auto dilation = toIValue(node->input(4))->toIntList();
      for (auto it = dilation.begin(); it != dilation.end(); ++it) {
        if ((*it) != 1) {
          // Does not support dilation is not 1 cases
          continue;
        }
      }

      auto newNode = replaceOpWithDNNL(node, block->owningGraph());
      newNode->appendReorder(natureFormat);
    } else if (node->matches("aten::avg_pool2d(Tensor self, int[] kernel_size, int[] stride=[], int[] padding, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor")) {
      auto newNode = replaceOpWithDNNL(node, block->owningGraph());
      newNode->appendReorder(natureFormat);
    }
  }
}

void OpRewritePass(std::shared_ptr<Graph>& graph) {
  OpRewritePass(graph->block());
  ConstantPropagation(graph);
}
}} // namespace torch::jit
