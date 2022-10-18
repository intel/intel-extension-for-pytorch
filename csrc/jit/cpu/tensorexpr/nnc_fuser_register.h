#pragma once

#include <torch/csrc/jit/api/module.h>

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

TORCH_API void registerCustomOp2NncFuser();

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
