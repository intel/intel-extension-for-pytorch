#include "prepare_binary.h"
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/shape_analysis.h>

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

using namespace torch::jit;

bool compareConstValue(Value* v, double d) {
  auto ival = toIValue(v);
  return ival.has_value() &&
      ((ival->isInt() && ival->toInt() == static_cast<int>(d)) ||
       (ival->isDouble() && ival->toDouble() == d));
}

void mayConvertScalarInputToTensor(Node* node) {
  // We do not handle binary ops with two scalar inputs,
  // and we assume scalar is always at the second place.
  if (node->input(0)->type()->isSubtypeOf(TensorType::get()) &&
      (node->input(1)->type()->isSubtypeOf(FloatType::get()) ||
       node->input(1)->type()->isSubtypeOf(IntType::get()))) {
    auto scalar = node->input(1);
    WithInsertPoint guard(node);
    auto g = node->owningGraph();
    // 42 : Scalar  -->  tensor(42.0) : Float([])
    auto t = g->insert(
        aten::as_tensor, {scalar}, {{"dtype", at::ScalarType::Float}});
    // tensor(42.0) : Float([])  -->  tensor([42.0]) : Float([1])
    c10::optional<size_t> t_dim = 1;
    auto target_type = TensorTypePtr(
        TensorType::create(at::ScalarType::Float, at::kCPU, t_dim, false));
    target_type = target_type->withSizes({1});
    t->setType(target_type);
    auto unsqueezed = g->insert(aten::unsqueeze, {t, 0});
    unsqueezed->setType(target_type);
    node->replaceInput(1, unsqueezed);
    // Add a mark here and convert tensor back to scalar later on for unfused
    // add/div
    node->i_(Symbol::attr("scalar"), true);
  }
}

void mayConvertTensorToScalarInput(Node* node) {
  if (node->numAttributes() == 0) {
    return;
  }
  TORCH_CHECK(
      node->hasAttributeS("scalar"),
      "add or div node with numAttributes != 0 must have attr: scalar");

  auto unsqueeze_node = node->input(1)->node();
  auto as_tensor_node = unsqueeze_node->input(0)->node();
  auto scalar_value = as_tensor_node->input(0);
  node->replaceInput(1, scalar_value);

  node->removeAttributeS("scalar");
}

static void ConvertScalarToTensor(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      ConvertScalarToTensor(sub);
    }

    if (node->kind() == aten::add || node->kind() == aten::div) {
      mayConvertScalarInputToTensor(node);
    }
  }
}

static void ConvertTensorToScalar(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      ConvertTensorToScalar(sub);
    }

    if (node->kind() == aten::add || node->kind() == aten::div) {
      mayConvertTensorToScalarInput(node);
    }
  }
}

void mayDecomposeAdd(Node* node) {
  if (node->inputs().size() < 3)
    return; // aten::add(int, int) may have only two inputs

  auto alphaEqualsOne = compareConstValue(node->input(2), 1.0);
  if (!alphaEqualsOne) {
    WithInsertPoint guard(node);
    auto g = node->owningGraph();
    auto mul = g->insert(aten::mul, {node->input(1), node->input(2)});
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

    if ((node->kind() == aten::add && compareConstValue(node->input(1), 0.0)) ||
        (node->kind() == aten::mul && compareConstValue(node->input(1), 1.0)) ||
        (node->kind() == aten::div && compareConstValue(node->input(1), 1.0))) {
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
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
