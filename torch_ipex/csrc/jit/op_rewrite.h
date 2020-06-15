#pragma once

#include <memory>
#include <ideep.hpp>
#include <torch/csrc/jit/ir/ir.h>

namespace torch { namespace jit {

void OpRewritePass(std::shared_ptr<Graph>& graph);

}} // namespace torch::jit
