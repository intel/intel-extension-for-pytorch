#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <memory>

namespace torch_ipex {
namespace jit {

void FusionPass(std::shared_ptr<torch::jit::Graph>& graph);
void ApplyInplaceOptimization(std::shared_ptr<torch::jit::Graph>& graph);
void IPEXFusionPass(std::shared_ptr<torch::jit::Graph>& graph);
void FoldPrepackingOps(torch::jit::script::Module& m);

} // namespace jit
} // namespace torch_ipex
