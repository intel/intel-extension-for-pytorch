#pragma once

#include <Macros.h>
#include <torch/csrc/jit/api/module.h>

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

IPEX_API void clearCustomOp2NncFuser();
IPEX_API void registerCustomOp2NncFuser();

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
