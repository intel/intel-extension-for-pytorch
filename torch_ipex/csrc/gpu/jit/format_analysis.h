#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch { namespace jit {
//
// This pass is for reorder elimination only
// Might incorporate more functionality in future
//
TORCH_API void FormatOptimize(std::shared_ptr<Graph>& graph);
}} // namespace torch::jit
