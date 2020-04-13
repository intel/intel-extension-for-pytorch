#pragma once

#include <memory>
#include <torch/csrc/jit/ir.h>

namespace torch { namespace jit {
void FusionPass(std::shared_ptr<Graph>& graph);
void InitFusionPass();
}} // namespace torch::jit
