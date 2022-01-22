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

#include "csrc/aten/cpu/WeightPack.h"
#include "folding_common_utils.h"
#include "frozen_linear_folding.h"

namespace torch {
namespace jit {

namespace {

using Tensor = at::Tensor;

bool supportedLinearNode(Node* n) {
  if (n->kind() == aten::linear ||
      n->kind() == Symbol::fromQualString("torch_ipex::ipex_linear")) {
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
    if (linear->kind() == aten::linear) {
      output_channel =
          constant_as<Tensor>(linear->namedInput("weight")).value().size(0);
    } else {
      output_channel =
          constant_as<int64_t>(linear->namedInput("out_features")).value();
    }
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

void FoldFrozenLinearAddOrSub(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      FoldFrozenLinearAddOrSub(block);
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
      if (linear->kind() == aten::linear) {
        add_or_sub_tensor = resizeConstantScalarOrTensorToShape(
            add_or_sub->inputs().at(1),
            {weight_tensor.size(0)},
            weight_tensor.options());
      } else {
        add_or_sub_tensor = resizeConstantScalarOrTensorToShape(
            add_or_sub->inputs().at(1),
            {constant_as<int64_t>(linear->namedInput("out_features")).value()},
            weight_tensor.options());
      }
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
      // DCE run after cleans up nodes
    }
  }
}

void FoldFrozenLinearMulOrDiv(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      FoldFrozenLinearMulOrDiv(block);
    }

    if (supportedMulOrDiv(n) &&
        supportedLinearNode(n->inputs().at(0)->node())) {
      auto linear = n->inputs().at(0)->node();
      auto mul_or_div = n;

      if (!checkLinearAndBroadcastingOpPreConditions(linear, mul_or_div)) {
        continue;
      }

      Tensor weight_tensor;
      if (linear->kind() == aten::linear) {
        weight_tensor =
            constant_as<Tensor>(linear->namedInput("weight")).value();
      } else {
        weight_tensor = torch_ipex::cpu::linear_weight_unpack(
            constant_as<Tensor>(linear->namedInput("weight")).value(),
            constant_as<int64_t>(linear->inputs().at(2)).value(),
            constant_as<int64_t>(linear->inputs().at(3)).value(),
            false,
            c10::nullopt);
      }

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

      Tensor fuse_weight;
      if (linear->kind() == aten::linear) {
        fuse_weight = (*stack_out)[0].toTensor().to(weight_tensor.dtype());
      } else {
        fuse_weight = torch_ipex::cpu::linear_weight_pack(
            (*stack_out)[0].toTensor().to(weight_tensor.dtype()), c10::nullopt);
      }

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

        fused_linear_weight->setDebugName(
            linear_b_value->debugName() + "_fused_" +
            mul_or_div->kind().toUnqualString());
        linear->replaceInputWith(linear_b_value, fused_linear_bias);
      }
      // DCE run after cleans up nodes
    }
  }
}

} // namespace

void FoldFrozenLinearAddOrSub(std::shared_ptr<Graph>& graph) {
  FoldFrozenLinearAddOrSub(graph->block());
  EliminateDeadCode(graph);
}

void FoldFrozenLinearMulOrDiv(std::shared_ptr<Graph>& graph) {
  FoldFrozenLinearMulOrDiv(graph->block());
  EliminateDeadCode(graph);
}

} // namespace jit
} // namespace torch
