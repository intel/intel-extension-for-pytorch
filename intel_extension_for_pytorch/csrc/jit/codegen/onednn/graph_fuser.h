#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

void CreateLlgaSubgraphs(std::shared_ptr<torch::jit::Graph>& graph);

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
