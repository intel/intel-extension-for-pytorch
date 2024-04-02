#include "prepare_binary.h"
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include "../LlgaTensorImpl.h"
#include "utils.h"

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

using namespace torch::jit;
using data_type = dnnl::graph::logical_tensor::data_type;

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
      // This tensor won't be added to oneDNN graph due to unsupported data
      // type, so no need to do promotion for it.
      if (getLlgaDataType(promotedDtype) == data_type::undef)
        return;
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
          // This tensor won't be added to oneDNN graph due to unsupported data
          // type, so no need to do promotion for it.
          if (getLlgaDataType(promotedDtype) == data_type::undef)
            return;
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
          // This tensor won't be added to oneDNN graph due to unsupported data
          // type, so no need to do promotion for it.
          if (getLlgaDataType(dtypeOfFirstInput) == data_type::undef)
            return;
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

void replaceWithSelectOpNode(Node* node, Value* unexpanded_mask) {
  WithInsertPoint guard(node);
  auto g = node->owningGraph();
  auto to_replace = node->input(1);
  auto dtype = node->input(0)->type()->cast<TensorType>()->scalarType().value();
  auto to_replace_tensor =
      g->insert(aten::as_tensor, {to_replace}, {{"dtype", dtype}});
  c10::optional<size_t> t_dim = 1;
  auto target_type =
      TensorTypePtr(TensorType::create(dtype, at::kCPU, t_dim, false));
  target_type = target_type->withSizes({1});
  to_replace_tensor->setType(target_type);
  auto unsqueezed = g->insert(aten::unsqueeze, {to_replace_tensor, 0});
  unsqueezed->setType(target_type);
  auto selectNode = g->insert(
      Symbol::fromQualString("llga::Select"),
      {unexpanded_mask, unsqueezed, node->input(0)});
  selectNode->setType(node->outputs()[0]->type());
  node->outputs()[0]->replaceAllUsesWith(selectNode);
}

static void replaceWithSelectOp(Block* block) {
  static auto registry = torch::RegisterOperators().op(
      "llga::Select(Tensor mask, Tensor then_tensor, Tensor else_tensor) -> Tensor");
  Value* unexpanded_mask = nullptr;
  for (auto nodeIterator = block->nodes().begin();
       nodeIterator != block->nodes().end();
       ++nodeIterator) {
    Node* node = *nodeIterator;
    for (auto blockIterator = node->blocks().begin();
         blockIterator != node->blocks().end();
         ++blockIterator) {
      Block* body_block = *blockIterator;
      replaceWithSelectOp(body_block);
    }
    if ((node->kind() == aten::masked_fill) && (unexpanded_mask != nullptr)) {
      replaceWithSelectOpNode(node, unexpanded_mask);
      nodeIterator.destroyCurrent();
    } else if (
        (node->kind() == aten::expand_as) &&
        (node->next()->kind() == aten::masked_fill) &&
        ((uintptr_t)(node->input(1)) == (uintptr_t)(node->next()->input(0)))) {
      unexpanded_mask = node->input(0);
      node->next()->removeInput(1);
      nodeIterator.destroyCurrent();
    }
  }
}

void removeSelectOpNode(Node* node) {
  WithInsertPoint guard(node);
  auto g = node->owningGraph();
  auto dtype = node->output()->type()->cast<TensorType>()->scalarType().value();
  // The sequence of ops in the graph is like this -
  // if_tensor = aten::as_tensor(%if_value, %, %)
  // if_tensor = aten::unsqueeze(%if_tensor, %57)
  // llga::Select(mask, if_tensor, then_tensor)
  auto as_tensor_node = node->input(1)->node()->input(0)->node();
  auto expand_as_output =
      g->insert(aten::expand_as, {node->input(0), node->input(2)});
  expand_as_output->setType(node->input(2)->type());
  auto masked_fill_output = g->insert(
      aten::masked_fill,
      {node->input(2), expand_as_output, as_tensor_node->input(0)});
  masked_fill_output->setType(node->input(2)->type());
  node->output()->replaceAllUsesWith(masked_fill_output);
}

static void mayRemoveLLGASelect(Block* block) {
  for (auto nodeIterator = block->nodes().begin();
       nodeIterator != block->nodes().end();
       ++nodeIterator) {
    Node* node = *nodeIterator;
    for (auto blockIterator = node->blocks().begin();
         blockIterator != node->blocks().end();
         ++blockIterator) {
      Block* body_block = *blockIterator;
      mayRemoveLLGASelect(body_block);
    }
    if (node->kind().toQualString() == std::string("llga::Select")) {
      removeSelectOpNode(node);
      nodeIterator.destroyCurrent();
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
  replaceWithSelectOp(graph->block());
  ConvertScalarToTensor(graph->block());
  // TODO: after conv-bn folding, bias will become bias? (Optional) after this
  // pass and will lose it when using mustNotBeNone to check Optional Bias
  // PropagateInputShapes(graph);
}

void RevertPrepareBinaryForLLGA(const std::shared_ptr<Graph>& graph) {
  ConvertTensorToScalar(graph->block());
  mayRevertDtypeAttributeInsertion(graph->block());
  mayRemoveLLGASelect(graph->block());
  EliminateDeadCode(graph);
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
