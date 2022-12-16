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

} // namespace utils
} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
