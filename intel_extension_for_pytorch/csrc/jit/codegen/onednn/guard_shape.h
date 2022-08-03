#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

void prepareFusionGroupAndGuardOutputs(torch::jit::Block* block);

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
