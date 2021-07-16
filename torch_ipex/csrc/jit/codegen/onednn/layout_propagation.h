#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// The rule is that LlgaTensor can only be consumed by JIT-only ops:
// e.g. llga fusion ops, prim ops (torch/csrc/jit/runtime/register_prim_ops.cpp).
// If a LlgaPartition is only fed to JIT-only ops, 
// the output format of this partition will be set as ANY.
void PropagateLayout(const std::shared_ptr<Graph>& graph);

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch