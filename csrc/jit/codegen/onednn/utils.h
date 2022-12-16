#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {
namespace utils {

bool isViewOp(torch::jit::Node* n);

bool isEltwiseOp(torch::jit::Node* n);

bool isSupportedAsInputToDequant(torch::jit::Node* n);

std::vector<int64_t> IntZeroDimTensorToVector(const at::Tensor& tensor);

double getScale(torch::jit::Node* input_node);

std::vector<int64_t> getZPSVector(torch::jit::Node* input_node);

} // namespace utils
} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
