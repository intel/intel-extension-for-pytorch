#pragma once

#include <torch/csrc/jit/ir/alias_analysis.h>

namespace torch {
namespace jit {

void RemoveRedundantAliases(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
