#include "utils.h"
#include "operator.h"

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {
namespace utils {

using namespace torch::jit;

bool isViewOp(Node* n) {
  switch (n->kind()) {
    case aten::view:
    case aten::permute:
    case aten::transpose:
      return true;
    default:
      return false;
  }
}

bool isBinaryOp(torch::jit::Node* n) {
  switch (n->kind()) {
    case aten::add:
    case aten::div:
    case aten::mul:
    case aten::max:
      return true;
    default:
      return false;
  }
}

bool isEltwiseOp(Node* n) {
  if (n->kind() == Symbol::aten("relu") ||
      n->kind() == Symbol::aten("sigmoid") ||
      n->kind() == Symbol::aten("quantize_per_tensor") ||
      n->kind() == Symbol::aten("quantize_per_channel") ||
      n->kind() == aten::to) {
    return true;
  } else {
    return false;
  }
}

bool isSupportedAsInputToDequant(torch::jit::Node* n) {
  if (n->kind() == prim::Constant ||
      n->kind() == Symbol::aten("quantize_per_tensor") ||
      n->kind() == Symbol::aten("quantize_per_channel")) {
    return true;
  } else {
    return false;
  }
}

std::vector<int64_t> IntZeroDimTensorToVector(const at::Tensor& tensor) {
  std::vector<int64_t> returnedVector;
  returnedVector.push_back(tensor.item().toInt());
  return returnedVector;
}

std::vector<int64_t> getZPSVector(Node* input_node) {
  std::vector<int64_t> zps_vector;
  auto zps_value = input_node->input(2);
  if (zps_value->type()->isSubtypeOf(TensorType::get())) {
    // Composing FX with JIT tracing may cause zps to be a 0-dim tensor
    zps_vector =
        IntZeroDimTensorToVector(toIValue(zps_value).value().toTensor());
  } else {
    // must be an int
    TORCH_CHECK(zps_value->type()->cast<IntType>(), "zps must be Int type");
    zps_vector = Operator::IntToVector(input_node, 2);
  }
  return zps_vector;
}

double getScale(Node* input_node) {
  double scale;
  auto scale_value = input_node->input(1);
  if (scale_value->type()->isSubtypeOf(TensorType::get())) {
    // Composing FX with JIT tracing may cause scale to be a 0-dim tensor
    scale = toIValue(scale_value).value().toTensor().item().toFloat();
  } else {
    TORCH_CHECK(
        scale_value->type()->cast<FloatType>(), "scale must be Float type");
    scale = Operator::Float(input_node, 1);
  }
  return scale;
}

bool isZeroPointSupported(Value* zps) {
  auto zps_value = toIValue(zps);
  return (
      zps_value.has_value() &&
      (zps_value->isInt() ||
       (zps_value->isTensor() &&
        (zps_value.value().toTensor().scalar_type() == at::ScalarType::Long))));
}

bool isScaleSupported(Value* scale) {
  auto scale_value = toIValue(scale);
  return (
      scale_value.has_value() &&
      (scale_value->isDouble() ||
       (scale_value->isTensor() &&
        (scale_value.value().toTensor().scalar_type() ==
         at::ScalarType::Float))));
}

bool compareConstValue(torch::jit::Value* v, double d) {
  auto ival = toIValue(v);
  return ival.has_value() &&
      ((ival->isInt() && ival->toInt() == static_cast<int>(d)) ||
       (ival->isDouble() && ival->toDouble() == d));
}

// Mark original dtype of a node before type-promotion, so that if the node
// would not be lowered to LLGA, then the change can be reverted.
void mark_original_output_dtype(torch::jit::Node* node) {
  auto outputDtype = node->output()->type()->expect<TensorType>()->scalarType();
  if (outputDtype.has_value()) {
    switch (outputDtype.value()) {
      case at::ScalarType::Float:
        node->i_(Symbol::attr("was_float"), true);
        break;
      case at::ScalarType::BFloat16:
        node->i_(Symbol::attr("was_bfloat16"), true);
        break;
      case at::kInt:
        node->i_(Symbol::attr("was_int"), true);
        break;
      default:
        break;
    }
  }
}

void convertInputTo0DTensor(
    torch::jit::Node* node,
    int input_index,
    at::ScalarType dtype) {
  mark_original_output_dtype(node);
  auto scalar = node->input(input_index);
  WithInsertPoint guard(node);
  auto g = node->owningGraph();
  // 42 : Scalar  -->  tensor(42.0) : Float([])
  auto scalar_tensor = g->insert(aten::as_tensor, {scalar}, {{"dtype", dtype}});
  auto target_type =
      TensorTypePtr(TensorType::create(dtype, at::kCPU, {}, false));
  scalar_tensor->setType(target_type);
  node->replaceInput(input_index, scalar_tensor);
  // Add a mark here and convert tensor back to scalar later on for unfused
  // add/div and some other binary ops
  node->i_(Symbol::attr("scalar"), true);
}

void modifyDtypeOfNode(torch::jit::Node* node, at::ScalarType dtype) {
  auto existingDtype =
      node->outputs()[0]->type()->expect<TensorType>()->scalarType();
  if (existingDtype.has_value()) {
    switch (existingDtype.value()) {
      case at::ScalarType::Float:
      case at::ScalarType::BFloat16:
      case at::kInt:
        node->outputs()[0]->setType(
            node->outputs()[0]->type()->expect<TensorType>()->withScalarType(
                dtype));
        break;
      default:
        break;
    }
  }
}

void insertTypeCast(
    torch::jit::Node* node,
    int input_index,
    at::ScalarType dtype) {
  WithInsertPoint guard(node);
  auto g = node->owningGraph();
  auto to_node_output =
      g->insert(aten::to, {node->input(input_index)}, {{"dtype", dtype}});
  to_node_output->setType(node->input(input_index)
                              ->type()
                              ->expect<TensorType>()
                              ->withScalarType(dtype));
  node->replaceInput(input_index, to_node_output);
}

void mayModifyOutputDtype(torch::jit::Node* node) {
  if (node->outputs()[0]->type()->isSubtypeOf(TensorType::get())) {
    if (node->hasAttributeS("was_float")) {
      modifyDtypeOfNode(node, at::ScalarType::Float);
      node->removeAttributeS("was_float");
    } else if (node->hasAttributeS("was_bfloat16")) {
      modifyDtypeOfNode(node, at::ScalarType::BFloat16);
      node->removeAttributeS("was_bfloat16");
    } else if (node->hasAttributeS("was_int")) {
      modifyDtypeOfNode(node, at::ScalarType::Int);
      node->removeAttributeS("was_int");
    }
  }
}

} // namespace utils
} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
