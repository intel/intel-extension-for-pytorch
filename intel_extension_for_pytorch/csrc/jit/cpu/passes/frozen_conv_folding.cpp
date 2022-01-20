#include <ATen/Functions.h>
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
#include "frozen_conv_folding.h"

namespace torch {
namespace jit {

namespace {

using Tensor = at::Tensor;

bool nonConstantParameters(Node* n) {
  // Checks if the parameters, not including the
  // first param are all constants.
  for (size_t i = 1; i < n->inputs().size(); i++) {
    if (n->inputs().at(i)->node()->kind() != prim::Constant) {
      return true;
    }
  }
  return false;
}

bool supportedConvNode(Node* n) {
  if (n->kind() == aten::conv2d || n->kind() == aten::conv3d ||
      n->kind() == Symbol::fromQualString("torch_ipex::convolution_forward")) {
    return true;
  } else {
    return false;
  }
}

bool supportedAddOrSub(Node* n) {
  if (n->kind() == aten::add || n->kind() == aten::sub) {
    return true;
  } else {
    return false;
  }
}

// In order to fuse add/sub/mul/div with conv, the dimensions of its
// constant tensor must satisfy the following:
// - with resizing, broadcast to w/ weight/bias tensor shape
// - broadcast to the conv output shape
// It needs to have a shape that can resize to weight/bias
// tensor shape because we need to run the op with the conv
// weights/bias without changing their sizes.
// It needs to broadcast to the conv output shape so that we do
// accidentally change the shape of op output by pre-fusing it
// compared to eager.
// The only dimension value shared by weight/bias/conv output
// is they all contain a dim with value = channels-out. In the
// conv output tensor, this is in the second dimension,
// so the pointwise op tensor may have a second dimension of
// value == channels-out, but all the other dimensions have to be 1
bool opDoesNotBroadCastWithConv(Tensor& op_tensor, Tensor& weight_tensor) {
  if (op_tensor.ndimension() > weight_tensor.ndimension()) {
    return false;
  }
  for (int64_t i = op_tensor.ndimension() - 1; i >= 0; i--) {
    // channels-out dimension == weight_tensor.size(0)
    if (i == 1 && op_tensor.size(i) == weight_tensor.size(0)) {
      continue;
    }
    if (op_tensor.size(i) != 1) {
      return false;
    }
  }
  return true;
}

bool checkConvAndBroadcastingOpPreConditions(Node* conv, Node* op) {
  if (nonConstantParameters(conv) || nonConstantParameters(op)) {
    return false;
  }

  if (conv->output()->uses().size() > 1) {
    return false;
  }

  Tensor weight_tensor =
      constant_as<Tensor>(conv->namedInput("weight")).value();

  // avoid fusing op that causes type promotion
  // resticting to float avoids int/float difficulties with scalar overload
  if (!weight_tensor.is_floating_point()) {
    return false;
  }

  if (op->inputs().at(1)->type()->cast<TensorType>()) {
    auto op_tensor = constant_as<Tensor>(op->inputs().at(1)).value();
    if (conv->kind() == aten::conv2d || conv->kind() == aten::conv3d) {
      if (!opDoesNotBroadCastWithConv(op_tensor, weight_tensor)) {
        return false;
      }
    } else {
      if (op_tensor.ndimension() > conv->inputs()
                                       .at(0)
                                       ->type()
                                       ->cast<TensorType>()
                                       ->sizes()
                                       .size()
                                       .value()) {
        return false;
      }
      for (int64_t i = op_tensor.ndimension() - 1; i >= 0; i--) {
        if (i == 1 &&
            op_tensor.size(i) ==
                constant_as<int64_t>(conv->namedInput("output_channel"))
                    .value()) {
          continue;
        }
        if (op_tensor.size(i) != 1) {
          return false;
        }
      }
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

Tensor resizeConstantScalarOrTensorToShape(
    Value* v,
    const std::vector<int64_t>& shape,
    at::TensorOptions options) {
  Tensor ret_tensor;
  if (v->type()->cast<TensorType>()) {
    ret_tensor = constant_as<Tensor>(v).value();
  } else {
    ret_tensor = at::zeros(shape, options);
    if (v->type()->cast<IntType>()) {
      ret_tensor.fill_(constant_as<int64_t>(v).value());
    } else {
      ret_tensor.fill_(constant_as<double>(v).value());
    }
  }

  if (ret_tensor.numel() == 1) {
    // expand errors if the shape input has less # dims than the tensor input
    ret_tensor = ret_tensor.reshape({1});
    ret_tensor = ret_tensor.expand(shape);
  } else {
    TORCH_INTERNAL_ASSERT(ret_tensor.numel() == c10::multiply_integers(shape));
    ret_tensor = ret_tensor.view(shape);
  }
  return ret_tensor;
}

void FoldFrozenConvAddOrSub(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      FoldFrozenConvAddOrSub(block);
    }

    if (supportedAddOrSub(n) && supportedConvNode(n->inputs().at(0)->node())) {
      auto conv = n->inputs().at(0)->node();
      auto add_or_sub = n;

      if (!checkConvAndBroadcastingOpPreConditions(conv, add_or_sub)) {
        continue;
      }

      Tensor weight_tensor =
          constant_as<Tensor>(conv->namedInput("weight")).value();

      Tensor add_or_sub_tensor;
      if (conv->kind() == aten::conv2d || conv->kind() == aten::conv3d) {
        add_or_sub_tensor = resizeConstantScalarOrTensorToShape(
            add_or_sub->inputs().at(1),
            {weight_tensor.size(0)},
            weight_tensor.options());
      } else {
        add_or_sub_tensor = resizeConstantScalarOrTensorToShape(
            add_or_sub->inputs().at(1),
            {constant_as<int64_t>(conv->namedInput("output_channel")).value()},
            weight_tensor.options());
      }
      Tensor bias;
      if (conv->namedInput("bias")->type() == NoneType::get()) {
        bias = at::zeros_like(add_or_sub_tensor, weight_tensor.dtype());
      } else {
        bias = constant_as<Tensor>(conv->namedInput("bias")).value();
      }

      WithInsertPoint guard(conv);

      add_or_sub->replaceInputWith(
          conv->output(), b->owningGraph()->insertConstant(bias));
      add_or_sub->replaceInput(
          1, b->owningGraph()->insertConstant(add_or_sub_tensor));

      auto stack_out = runNodeIfInputsAreConstant(add_or_sub);
      TORCH_INTERNAL_ASSERT(stack_out && stack_out->size() == 1);
      Tensor fuse_bias = (*stack_out)[0].toTensor().to(bias.dtype());

      auto fused_conv_b = b->owningGraph()->insertConstant(fuse_bias);
      auto conv_b_value = conv->namedInput("bias");

      fused_conv_b->setDebugName(
          conv_b_value->debugName() + "_fused_" +
          add_or_sub->kind().toUnqualString());
      conv->replaceInputWith(conv_b_value, fused_conv_b);
      add_or_sub->output()->replaceAllUsesWith(conv->output());
      // DCE run after cleans up nodes
    }
  }
}

bool supportedMulOrDiv(Node* n) {
  if (n->kind() == aten::mul || n->kind() == aten::div) {
    return true;
  } else {
    return false;
  }
}

void FoldFrozenConvMulOrDiv(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      FoldFrozenConvMulOrDiv(block);
    }

    if (supportedMulOrDiv(n) && supportedConvNode(n->inputs().at(0)->node())) {
      auto conv = n->inputs().at(0)->node();
      auto mul_or_div = n;

      if (!checkConvAndBroadcastingOpPreConditions(conv, mul_or_div)) {
        continue;
      }

      Tensor weight_tensor;
      if (conv->kind() == aten::conv2d || conv->kind() == aten::conv3d) {
        weight_tensor = constant_as<Tensor>(conv->namedInput("weight")).value();
      } else {
        weight_tensor = torch_ipex::cpu::convolution_weight_unpack(
            constant_as<Tensor>(conv->namedInput("weight")).value(),
            toIValue(conv->namedInput("padding"))->toIntVector(),
            toIValue(conv->namedInput("stride"))->toIntVector(),
            toIValue(conv->namedInput("dilation"))->toIntVector(),
            toIValue(conv->namedInput("kernel_size"))->toIntVector(),
            constant_as<int64_t>(conv->namedInput("groups")).value(),
            constant_as<int64_t>(conv->namedInput("output_channel")).value(),
            conv->inputs()
                .at(0)
                ->type()
                ->cast<TensorType>()
                ->sizes()
                .concrete_sizes()
                .value()[1],
            constant_as<bool>(conv->namedInput("weight_channels_last")).value(),
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

      WithInsertPoint guard(conv);

      Tensor mul_tensor = resizeConstantScalarOrTensorToShape(
          mul_or_div->inputs().at(1),
          weight_compatible_size,
          weight_tensor.options());

      // First fold with weight tensor
      mul_or_div->replaceInputWith(
          conv->output(), b->owningGraph()->insertConstant(weight_tensor));
      mul_or_div->replaceInput(1, b->owningGraph()->insertConstant(mul_tensor));

      auto stack_out = runNodeIfInputsAreConstant(mul_or_div);
      TORCH_INTERNAL_ASSERT(stack_out && stack_out->size() == 1);

      Tensor fuse_weight;
      if (conv->kind() == aten::conv2d || conv->kind() == aten::conv3d) {
        fuse_weight = (*stack_out)[0].toTensor().to(weight_tensor.dtype());
      } else {
        fuse_weight = torch_ipex::cpu::convolution_weight_pack(
            (*stack_out)[0].toTensor().to(weight_tensor.dtype()),
            toIValue(conv->namedInput("padding"))->toIntVector(),
            toIValue(conv->namedInput("stride"))->toIntVector(),
            toIValue(conv->namedInput("dilation"))->toIntVector(),
            constant_as<int64_t>(conv->namedInput("groups")).value(),
            c10::nullopt);
      }

      auto fused_conv_weight = b->owningGraph()->insertConstant(fuse_weight);
      auto conv_weight_value = conv->namedInput("weight");

      fused_conv_weight->setDebugName(
          conv_weight_value->debugName() + "_fused_" +
          mul_or_div->kind().toUnqualString());
      conv->replaceInputWith(conv_weight_value, fused_conv_weight);
      mul_or_div->output()->replaceAllUsesWith(conv->output());

      // now fold with bias tensor
      if (conv->namedInput("bias")->type() != NoneType::get()) {
        Tensor bias = constant_as<Tensor>(conv->namedInput("bias")).value();
        // bias is of shape {channels_out}
        auto mul_tensor = resizeConstantScalarOrTensorToShape(
            mul_or_div->inputs().at(1), {out_channels}, bias.options());

        mul_or_div->replaceInput(0, b->owningGraph()->insertConstant(bias));
        mul_or_div->replaceInput(
            1, b->owningGraph()->insertConstant(mul_tensor));

        auto stack_out = runNodeIfInputsAreConstant(mul_or_div);
        TORCH_INTERNAL_ASSERT(stack_out && stack_out->size() == 1);
        Tensor fuse_bias = (*stack_out)[0].toTensor().to(bias.dtype());

        auto fused_conv_bias = b->owningGraph()->insertConstant(fuse_bias);
        auto conv_b_value = conv->namedInput("bias");

        fused_conv_weight->setDebugName(
            conv_b_value->debugName() + "_fused_" +
            mul_or_div->kind().toUnqualString());
        conv->replaceInputWith(conv_b_value, fused_conv_bias);
      }
      // DCE run after cleans up nodes
    }
  }
}

} // namespace

void FoldFrozenConvAddOrSub(std::shared_ptr<Graph>& graph) {
  FoldFrozenConvAddOrSub(graph->block());
  EliminateDeadCode(graph);
}

void FoldFrozenConvMulOrDiv(std::shared_ptr<Graph>& graph) {
  FoldFrozenConvMulOrDiv(graph->block());
  EliminateDeadCode(graph);
}

} // namespace jit
} // namespace torch
