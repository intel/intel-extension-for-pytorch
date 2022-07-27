#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {
namespace utils {

bool isViewOp(torch::jit::Node* n);

bool isEltwiseOp(torch::jit::Node* n);

} // namespace utils
} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
