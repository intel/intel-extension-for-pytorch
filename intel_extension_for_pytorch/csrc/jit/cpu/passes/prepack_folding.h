#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch_ipex {
namespace jit {

void PrePackingOpsFolder(std::shared_ptr<torch::jit::Graph>& graph);

} // namespace jit
} // namespace torch_ipex
