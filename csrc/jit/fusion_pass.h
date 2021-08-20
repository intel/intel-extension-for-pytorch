#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <memory>

namespace torch {
namespace jit {
// LEGACY CALL
struct RegisterPreFusionPass {
  RegisterPreFusionPass(GraphPass p);
};
} // namespace jit
} // namespace torch

namespace torch_ipex {
namespace jit {
using torch::jit::Graph;
void FusionPass(std::shared_ptr<Graph>& graph);
} // namespace jit
} // namespace torch_ipex
