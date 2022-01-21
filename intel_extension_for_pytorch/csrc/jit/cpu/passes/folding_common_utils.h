#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace jit {

inline bool nonConstantParameters(Node* n) {
  // Checks if the parameters, not including the
  // first param are all constants.
  for (size_t i = 1; i < n->inputs().size(); i++) {
    if (n->inputs().at(i)->node()->kind() != prim::Constant) {
      return true;
    }
  }
  return false;
}

inline bool supportedAddOrSub(Node* n) {
  if (n->kind() == aten::add || n->kind() == aten::sub) {
    return true;
  } else {
    return false;
  }
}

inline bool supportedMulOrDiv(Node* n) {
  if (n->kind() == aten::mul || n->kind() == aten::div) {
    return true;
  } else {
    return false;
  }
}

inline at::Tensor resizeConstantScalarOrTensorToShape(
    Value* v,
    const std::vector<int64_t>& shape,
    at::TensorOptions options) {
  at::Tensor ret_tensor;
  if (v->type()->cast<TensorType>()) {
    ret_tensor = constant_as<at::Tensor>(v).value();
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

} // namespace jit
} // namespace torch