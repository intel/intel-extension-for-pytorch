#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {


// Prepare dequant op for LLGA
//
// The pass decomposes the dequant node from:
//            quant
//      + - - - | - - - +
//      |    dequant    |
//      |    /     \    |
//      |  node1  node2 |
//      + - | - - - | - +
//              
// into:
//            quant
//      + - - / - \ - - +
//      |dequant dequant|
//      |    |      |   |
//      | node1 node2   |
//      + - | - - - | - +

void PrepareDequantForLLGA(std::shared_ptr<Graph>& graph);

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch