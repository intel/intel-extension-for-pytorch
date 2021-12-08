#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {
namespace utils {

bool isViewOp(Node* n);

bool isEltwiseOp(Node* n);

} // namespace utils
} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
