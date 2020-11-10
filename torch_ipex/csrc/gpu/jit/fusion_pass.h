#pragma once

#include <memory>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>

namespace torch { namespace jit {

// LEGACY CALL
struct TORCH_API RegisterPreFusionPass {
  RegisterPreFusionPass(GraphPass p);
};

void FusionPass(std::shared_ptr<Graph>& graph);
void InitFusionPass();
}} // namespace torch::jit
