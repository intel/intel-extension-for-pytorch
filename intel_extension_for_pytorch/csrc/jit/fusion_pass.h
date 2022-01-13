#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <memory>

namespace torch {
namespace jit {
void FusionPass(std::shared_ptr<Graph>& graph);
void ApplyInplaceOptimization(std::shared_ptr<Graph>& graph);
void IPEXFusionPass(std::shared_ptr<Graph>& graph);
void FoldPrepackingOps(script::Module& m);
} // namespace jit
} // namespace torch
