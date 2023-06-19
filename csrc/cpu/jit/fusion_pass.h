#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <memory>

namespace torch_ipex {
namespace jit {

TORCH_API void FusionPass(std::shared_ptr<torch::jit::Graph>& graph);
TORCH_API void ApplyInplaceOptimization(
    std::shared_ptr<torch::jit::Graph>& graph);
TORCH_API void IPEXFusionPass(std::shared_ptr<torch::jit::Graph>& graph);
TORCH_API void FoldPrepackingOps(torch::jit::script::Module& m);

} // namespace jit
} // namespace torch_ipex
