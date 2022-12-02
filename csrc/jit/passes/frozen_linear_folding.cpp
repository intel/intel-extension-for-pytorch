#include <ATen/Utils.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/tensorexpr/types.h>

#include "aten/WeightPack.h"
#include "folding_common_utils.h"
#include "frozen_linear_folding.h"

namespace torch_ipex {
namespace jit {

namespace graph_rewrite {

using namespace torch::jit;
using namespace torch_ipex::cpu;
using Tensor = at::Tensor;

bool supportedLinearNode(Node* n) {
  if (n->kind() == aten::linear ||
      n->kind() == Symbol::fromQualString("torch_ipex::ipex_linear") ||
      n->kind() == Symbol::fromQualString("torch_ipex::ipex_MKLSGEMM")) {
    return true;
  } else {
    return false;
  }
}

bool checkLinearAndBroadcastingOpPreConditions(Node* linear, Node* op) {
  if (nonConstantParameters(linear) || nonConstantParameters(op)) {
    return false;
  }

  if (linear->output()->uses().size() > 1) {
    return false;
  }

  Tensor weight_tensor =
      constant_as<Tensor>(linear->namedInput("weight")).value();

  // avoid fusing op that causes type promotion
  // resticting to float avoids int/float difficulties with scalar overload
  if (!weight_tensor.is_floating_point()) {
    return false;
  }

  if (op->inputs().at(1)->type()->cast<TensorType>()) {
    auto op_tensor = constant_as<Tensor>(op->inputs().at(1)).value();

    int64_t output_channel;
    output_channel =
        constant_as<Tensor>(linear->namedInput("weight")).value().size(0);
    if (op_tensor.sizes() != at::IntArrayRef({1, output_channel}) &&
        op_tensor.sizes() != at::IntArrayRef({output_channel})) {
      return false;
    }

    if (!op_tensor.is_floating_point() &&
        c10::promoteTypes(
            op_tensor.scalar_type(), weight_tensor.scalar_type()) !=
            weight_tensor.scalar_type()) {
      return false;
    }
  }

  return true;
}

std::tuple<at::Tensor, at::Tensor> computeUpdatedLinearWeightAndBias(
    const LinearBNParameters& p) {
  at::Tensor bn_scale = p.bn_w * at::rsqrt(p.bn_rv + p.bn_eps);
  at::Tensor fused_w = p.linear_w * bn_scale.unsqueeze(-1);
  at::Tensor fused_b = (p.linear_b - p.bn_rm) * bn_scale + p.bn_b;

  auto linear_w_dtype = p.linear_w.dtype();
  auto linear_b_dtype = p.linear_b.dtype();

  return std::make_tuple(
      fused_w.to(linear_w_dtype), fused_b.to(linear_b_dtype));
}

bool FoldFrozenLinearBatchnorm(Block* b) {
  bool graph_modified = false;
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      graph_modified |= FoldFrozenLinearBatchnorm(block);
    }

    if (n->kind() == aten::batch_norm &&
        supportedLinearNode(n->inputs().at(0)->node())) {
      auto linear = n->inputs().at(0)->node();
      auto bn = n;
      if (nonConstantParameters(linear) || nonConstantParameters(bn)) {
        continue;
      }

      auto bn_rm_ivalue = bn->namedInput("running_mean");
      auto bn_rv_ivalue = bn->namedInput("running_var");

      // check running_mean and running_var has value, if they are
      // None(track_running_stats=False), skiping the folding path.
      if (bn_rm_ivalue->type() == NoneType::get() &&
          bn_rv_ivalue->type() == NoneType::get()) {
        continue;
      }

      auto bn_rm = constant_as<Tensor>(bn->namedInput("running_mean")).value();
      auto bn_rv = constant_as<Tensor>(bn->namedInput("running_var")).value();
      auto bn_eps = constant_as<double>(bn->namedInput("eps")).value();
      auto linear_w = constant_as<Tensor>(linear->namedInput("weight")).value();

      // implementation taken from torch/nn/utils/fusion.py
      Tensor linear_b;
      if (linear->namedInput("bias")->type() == NoneType::get()) {
        linear_b = at::zeros_like(bn_rm);
      } else {
        linear_b = constant_as<Tensor>(linear->namedInput("bias")).value();
      }
      Tensor bn_w;
      if (bn->namedInput("weight")->type() == NoneType::get()) {
        bn_w = at::ones_like(bn_rm);
      } else {
        bn_w = constant_as<Tensor>(bn->namedInput("weight")).value();
      }
      Tensor bn_b;
      if (n->namedInput("bias")->type() == NoneType::get()) {
        bn_b = at::zeros_like(bn_rm);
      } else {
        bn_b = constant_as<Tensor>(bn->namedInput("bias")).value();
      }

      LinearBNParameters params;
      params.linear_w = linear_w;
      params.linear_b = linear_b;
      params.bn_rm = bn_rm;
      params.bn_rv = bn_rv;
      params.bn_eps = bn_eps;
      params.bn_w = bn_w;
      params.bn_b = bn_b;
      std::tuple<Tensor, Tensor> out =
          computeUpdatedLinearWeightAndBias(params);
      WithInsertPoint guard(linear);
      auto fused_linear_w = b->owningGraph()->insertConstant(std::get<0>(out));
      auto fused_linear_b = b->owningGraph()->insertConstant(std::get<1>(out));
      auto linear_w_value = linear->namedInput("weight");
      auto linear_b_value = linear->namedInput("bias");

      fused_linear_w->setDebugName(linear_w_value->debugName() + "_fused_bn");
      fused_linear_b->setDebugName(linear_b_value->debugName() + "_fused_bn");

      linear->replaceInputWith(linear_w_value, fused_linear_w);
      linear->replaceInputWith(linear_b_value, fused_linear_b);

      bn->output()->replaceAllUsesWith(linear->output());
      graph_modified = true;
    }
  }
  return graph_modified;
}

bool FoldFrozenLinearAddOrSub(Block* b) {
  bool graph_modified = false;
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      graph_modified |= FoldFrozenLinearAddOrSub(block);
    }

    if (supportedAddOrSub(n) &&
        supportedLinearNode(n->inputs().at(0)->node())) {
      auto linear = n->inputs().at(0)->node();
      auto add_or_sub = n;

      if (!checkLinearAndBroadcastingOpPreConditions(linear, add_or_sub)) {
        continue;
      }

      Tensor weight_tensor =
          constant_as<Tensor>(linear->namedInput("weight")).value();

      Tensor add_or_sub_tensor;
      add_or_sub_tensor = resizeConstantScalarOrTensorToShape(
          add_or_sub->inputs().at(1),
          {weight_tensor.size(0)},
          weight_tensor.options());
      Tensor bias;
      if (linear->namedInput("bias")->type() == NoneType::get()) {
        bias = at::zeros_like(add_or_sub_tensor, weight_tensor.dtype());
      } else {
        bias = constant_as<Tensor>(linear->namedInput("bias")).value();
      }

      WithInsertPoint guard(linear);

      add_or_sub->replaceInputWith(
          linear->output(), b->owningGraph()->insertConstant(bias));
      add_or_sub->replaceInput(
          1, b->owningGraph()->insertConstant(add_or_sub_tensor));

      auto stack_out = runNodeIfInputsAreConstant(add_or_sub);
      TORCH_INTERNAL_ASSERT(stack_out && stack_out->size() == 1);
      Tensor fuse_bias = (*stack_out)[0].toTensor().to(bias.dtype());
      auto fused_linear_b = b->owningGraph()->insertConstant(fuse_bias);
      auto linear_b_value = linear->namedInput("bias");

      fused_linear_b->setDebugName(
          linear_b_value->debugName() + "_fused_" +
          add_or_sub->kind().toUnqualString());
      linear->replaceInputWith(linear_b_value, fused_linear_b);
      add_or_sub->output()->replaceAllUsesWith(linear->output());
      graph_modified = true;
      // DCE run after cleans up nodes
    }
  }
  return graph_modified;
}

bool FoldFrozenLinearMulOrDiv(Block* b) {
  bool graph_modified = false;
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      graph_modified |= FoldFrozenLinearMulOrDiv(block);
    }

    if (supportedMulOrDiv(n) &&
        supportedLinearNode(n->inputs().at(0)->node())) {
      auto linear = n->inputs().at(0)->node();
      auto mul_or_div = n;

      if (!checkLinearAndBroadcastingOpPreConditions(linear, mul_or_div)) {
        continue;
      }

      c10::intrusive_ptr<LinearOpContext> linear_op_ctx;

      Tensor weight_tensor;
      weight_tensor = constant_as<Tensor>(linear->namedInput("weight")).value();

      int64_t out_channels = weight_tensor.size(0);

      // We've already verified that the second input has numel == 1 or
      // channels-out resize it to the shape that will broadcast to
      // weight_tensor when the op is run so we dont change weight size
      std::vector<int64_t> weight_compatible_size = {out_channels};
      for (const auto i : c10::irange(1, weight_tensor.ndimension())) {
        (void)i; // Suppress unused variable warning
        weight_compatible_size.push_back(1);
      }

      WithInsertPoint guard(linear);

      Tensor mul_tensor = resizeConstantScalarOrTensorToShape(
          mul_or_div->inputs().at(1),
          weight_compatible_size,
          weight_tensor.options());

      // First fold with weight tensor
      mul_or_div->replaceInputWith(
          linear->output(), b->owningGraph()->insertConstant(weight_tensor));
      mul_or_div->replaceInput(1, b->owningGraph()->insertConstant(mul_tensor));

      auto stack_out = runNodeIfInputsAreConstant(mul_or_div);
      TORCH_INTERNAL_ASSERT(stack_out && stack_out->size() == 1);

      Tensor fuse_weight = (*stack_out)[0].toTensor().to(weight_tensor.dtype());
      auto fused_linear_weight = b->owningGraph()->insertConstant(fuse_weight);
      auto linear_weight_value = linear->namedInput("weight");

      fused_linear_weight->setDebugName(
          linear_weight_value->debugName() + "_fused_" +
          mul_or_div->kind().toUnqualString());
      linear->replaceInputWith(linear_weight_value, fused_linear_weight);

      mul_or_div->output()->replaceAllUsesWith(linear->output());

      // now fold with bias tensor
      if (linear->namedInput("bias")->type() != NoneType::get()) {
        Tensor bias = constant_as<Tensor>(linear->namedInput("bias")).value();
        // bias is of shape {channels_out}
        auto mul_tensor = resizeConstantScalarOrTensorToShape(
            mul_or_div->inputs().at(1), {out_channels}, bias.options());

        mul_or_div->replaceInput(0, b->owningGraph()->insertConstant(bias));
        mul_or_div->replaceInput(
            1, b->owningGraph()->insertConstant(mul_tensor));

        auto stack_out = runNodeIfInputsAreConstant(mul_or_div);
        TORCH_INTERNAL_ASSERT(stack_out && stack_out->size() == 1);
        Tensor fuse_bias = (*stack_out)[0].toTensor().to(bias.dtype());
        auto fused_linear_bias = b->owningGraph()->insertConstant(fuse_bias);
        auto linear_b_value = linear->namedInput("bias");
        linear->replaceInputWith(linear_b_value, fused_linear_bias);
      }
      graph_modified = true;
      // DCE run after cleans up nodes
    }
  }
  return graph_modified;
}

bool FoldFrozenLinearBatchnorm(std::shared_ptr<Graph>& graph) {
  bool graph_modified = FoldFrozenLinearBatchnorm(graph->block());
  EliminateDeadCode(graph);
  return graph_modified;
}

bool FoldFrozenLinearAddOrSub(std::shared_ptr<Graph>& graph) {
  bool graph_modified = FoldFrozenLinearAddOrSub(graph->block());
  EliminateDeadCode(graph);
  return graph_modified;
}

bool FoldFrozenLinearMulOrDiv(std::shared_ptr<Graph>& graph) {
  bool graph_modified = FoldFrozenLinearMulOrDiv(graph->block());
  EliminateDeadCode(graph);
  return graph_modified;
}

void FrozenLinearFolding(std::shared_ptr<Graph>& graph) {
  // run a couple times to capture Conv -> Mul -> Add etc
  bool changed;
  do {
    changed = false;
    changed |= FoldFrozenLinearBatchnorm(graph);
    changed |= FoldFrozenLinearAddOrSub(graph);
    changed |= FoldFrozenLinearMulOrDiv(graph);
  } while (changed);
}

} // namespace graph_rewrite
} // namespace jit
} // namespace torch_ipex
