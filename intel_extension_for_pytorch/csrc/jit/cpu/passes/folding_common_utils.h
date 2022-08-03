#pragma once

#include <ATen/ATen.h>

namespace torch_ipex {
namespace jit {

inline bool nonConstantParameters(torch::jit::Node* n) {
  // Checks if the parameters, not including the
  // first param are all constants.
  for (size_t i = 1; i < n->inputs().size(); i++) {
    if (n->inputs().at(i)->node()->kind() != torch::jit::prim::Constant) {
      return true;
    }
  }
  return false;
}

inline bool supportedAddOrSub(torch::jit::Node* n) {
  if (n->kind() == torch::jit::aten::add ||
      n->kind() == torch::jit::aten::sub) {
    return true;
  } else {
    return false;
  }
}

inline bool supportedMulOrDiv(torch::jit::Node* n) {
  if (n->kind() == torch::jit::aten::mul ||
      n->kind() == torch::jit::aten::div) {
    return true;
  } else {
    return false;
  }
}

inline at::Tensor resizeConstantScalarOrTensorToShape(
    torch::jit::Value* v,
    const std::vector<int64_t>& shape,
    at::TensorOptions options) {
  at::Tensor ret_tensor;
  if (v->type()->cast<torch::jit::TensorType>()) {
    ret_tensor = torch::jit::constant_as<at::Tensor>(v).value();
  } else {
    ret_tensor = at::zeros(shape, options);
    if (v->type()->cast<torch::jit::IntType>()) {
      ret_tensor.fill_(torch::jit::constant_as<int64_t>(v).value());
    } else {
      ret_tensor.fill_(torch::jit::constant_as<double>(v).value());
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
} // namespace torch_ipex
