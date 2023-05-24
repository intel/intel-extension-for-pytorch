#include "prepare_binary.h"
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include "utils.h"

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

using namespace torch::jit;

void handleBinaryOpInputs(Node* node, int first_input, int second_input) {
  if (node->input(first_input)->type()->isSubtypeOf(TensorType::get()) &&
      node->input(first_input)
          ->type()
          ->cast<TensorType>()
          ->scalarType()
          .has_value()) {
    auto dtypeOfFirstInput = node->input(first_input)
                                 ->type()
                                 ->cast<TensorType>()
                                 ->scalarType()
                                 .value();
    if (node->input(second_input)->type()->isSubtypeOf(NumberType::get())) {
      // If a scalar is added to be a tensor, we would assume that the
      // scalar is of the same dtype as the tensor, as oneDNN graph
      // currently requires inputs of binary ops to have the same dtype.
      // We create a 0D tensor from the scalar input & "promote" its
      // dtype to that of the first input. Doing so helps us satisfy PyTorch's
      // type promotion rules listed at
      // clang-format off
      // https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc
      // clang-format on
      auto promotedDtype = dtypeOfFirstInput;
      utils::convertInputTo0DTensor(node, second_input, promotedDtype);
      // dtype might have changed, so needs to be updated in IR as well
      utils::modifyDtypeOfNode(node, promotedDtype);
    } else if (node->input(second_input)
                   ->type()
                   ->isSubtypeOf(TensorType::get())) {
      // Here, both inputs are tensors, and we just wanna make sure that they
      // are the same dtype, as oneDNN Graph requires both inputs to have the
      // same dtype. We'll follow PyTorch's type-promotion rules here.
      auto second_input_typeptr =
          node->input(second_input)->type()->expect<TensorType>();
      c10::optional<at::ScalarType> second_input_type =
          second_input_typeptr->scalarType();
      if (second_input_type.has_value()) {
        // dtype of the second tensor might not be available in the IR
        auto dtypeOfSecondInput = second_input_type.value();
        if (dtypeOfFirstInput != dtypeOfSecondInput) {
          // Type promotion is required
          auto promotedDtype =
              c10::promoteTypes(dtypeOfFirstInput, dtypeOfSecondInput);
          int input_to_replace;
          if (promotedDtype == dtypeOfFirstInput) {
            input_to_replace = second_input;
          } else {
            input_to_replace = first_input;
          }
          // insert aten::to for typecasting
          utils::insertTypeCast(node, input_to_replace, promotedDtype);
          // dtype might have changed, so needs to be updated in IR as well
          utils::mark_original_output_dtype(node);
          utils::modifyDtypeOfNode(node, promotedDtype);
        } else {
          // both dtypes are same
          // IR info of dtypes is missing sometimes in JIT IR,
          // and we shouldn't treat those tensors as FP32 tensors by default.
          utils::mark_original_output_dtype(node);
          utils::modifyDtypeOfNode(node, dtypeOfFirstInput);
        }
      } // end inner if block
    } // end outer if block
  }
}

static void ConvertScalarToTensor(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      ConvertScalarToTensor(sub);
    }

    if (utils::isBinaryOp(node)) {
      handleBinaryOpInputs(node, 0, 1);
    } else if (node->kind() == aten::where) {
      // special case for when input index 2 maybe scalar
      handleBinaryOpInputs(node, 1, 2);
    }
  }
}

void mayConvertTensorToScalarInput(Node* node, int index) {
  if (node->numAttributes() == 0) {
    return;
  } else if (node->hasAttributeS("scalar")) {
    auto as_tensor_node = node->input(index)->node();
    auto scalar_value = as_tensor_node->input(0);
    node->replaceInput(index, scalar_value);
    utils::mayModifyOutputDtype(node);
    node->removeAttributeS("scalar");
  }
}

void mayRevertDtypeAttributeInsertion(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      mayRevertDtypeAttributeInsertion(sub);
    }
    if (utils::isBinaryOp(node)) {
      utils::mayModifyOutputDtype(node);
    }
  }
}

static void ConvertTensorToScalar(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      ConvertTensorToScalar(sub);
    }

    if (utils::isBinaryOp(node)) {
      mayConvertTensorToScalarInput(node, 1);
    } else if (node->kind() == aten::where) {
      mayConvertTensorToScalarInput(node, 2);
    }
  }
}

void mayDecomposeAdd(Node* node) {
  if (node->inputs().size() < 3)
    return; // aten::add(int, int) may have only two inputs

  auto alphaEqualsOne = utils::compareConstValue(node->input(2), 1.0);
  if (!alphaEqualsOne) {
    WithInsertPoint guard(node);
    auto g = node->owningGraph();
    auto mul = g->insert(aten::mul, {node->input(1), node->input(2)});
    auto dtype = node->output()->type()->expect<TensorType>()->scalarType();
    auto target_type =
        TensorTypePtr(TensorType::create(dtype, at::kCPU, {}, false));
    mul->setType(target_type);

    node->replaceInput(1, mul);
    auto one = g->insertConstant(1.0);
    node->replaceInput(2, one);
  }
}

static void DecomposeFusedAdd(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      DecomposeFusedAdd(sub);
    }

    if (node->kind() == aten::add) {
      mayDecomposeAdd(node);
    }
  }
}

static void EliminateIdentityMulAddDiv(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      EliminateIdentityMulAddDiv(sub);
    }

    if ((node->kind() == aten::add &&
         utils::compareConstValue(node->input(1), 0.0)) ||
        (node->kind() == aten::mul &&
         utils::compareConstValue(node->input(1), 1.0)) ||
        (node->kind() == aten::div &&
         utils::compareConstValue(node->input(1), 1.0))) {
      node->output()->replaceAllUsesWith(node->input(0));
    }
  }
}

void PrepareBinaryForLLGA(const std::shared_ptr<Graph>& graph) {
  DecomposeFusedAdd(graph->block());
  EliminateIdentityMulAddDiv(graph->block());
  EliminateDeadCode(graph);
  // ConvertScalarToTensor must be placed after EliminateIdentityMulAddDiv
  ConvertScalarToTensor(graph->block());
  // TODO: after conv-bn folding, bias will become bias? (Optional) after this
  // pass and will lose it when using mustNotBeNone to check Optional Bias
  // PropagateInputShapes(graph);
}

void RevertPrepareBinaryForLLGA(const std::shared_ptr<Graph>& graph) {
  ConvertTensorToScalar(graph->block());
  mayRevertDtypeAttributeInsertion(graph->block());
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
