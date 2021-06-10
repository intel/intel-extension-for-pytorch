#pragma once

#include <memory>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>

namespace torch { namespace jit {
// LEGACY CALL
struct RegisterPreFusionPass {
  RegisterPreFusionPass(GraphPass p);
};
}} // namespace torch::jit


namespace torch_ipex { namespace jit {
using torch::jit::Graph;
void FusionPass(std::shared_ptr<Graph> &graph);

void InitFusionPass();
}} // namespace torch_ipex::jit
