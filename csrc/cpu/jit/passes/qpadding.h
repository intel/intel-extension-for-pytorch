#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include "graph_rewrite.h"
#include "graph_rewrite_utils.h"

namespace torch_ipex {
namespace jit {

void QPaddingConversion(std::shared_ptr<torch::jit::Graph>& graph);

} // namespace jit
} // namespace torch_ipex
