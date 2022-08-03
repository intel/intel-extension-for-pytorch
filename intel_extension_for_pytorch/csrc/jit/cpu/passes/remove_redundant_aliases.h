#pragma once

#include <torch/csrc/jit/ir/alias_analysis.h>

namespace torch_ipex {
namespace jit {

void RemoveRedundantAliases(const std::shared_ptr<torch::jit::Graph>& graph);

} // namespace jit
} // namespace torch_ipex
