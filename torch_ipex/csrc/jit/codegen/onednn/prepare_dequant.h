#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {


// Prepare dequant op for LLGA
//
// The pass decomposes the dequant node from:
// graph 1:
//            quant
//      + - - - | - - - +
//      |    dequant    |
//      |    /     \    |
//      |  node1  node2 |
//      + - | - - - | - +
//       quant   quant    
// into:
// graph 2:
//            quant
//      + - - / - \ - - +
//      |dequant dequant|
//      |    |      |   |
//      | node1 node2   |
//      + - | - - - | - +
//       quant   quant
//
// In graph 1, the dequant node is shared by node1 and node2,
// as a result, neither node1 nor node2 could form an int8
// fusion pattern.
// After the decomposition, the graph 2 could hit the int8
// fusion pattern: dequant-node-quant, respectively for
// node1 and node2.
void PrepareDequantForLLGA(std::shared_ptr<Graph>& graph);

// PyTorch dequant node receives qtensor as input, thus no quantization-related
// info (scales, zp, etc.) on the IR, while LLGA needs those info on the
// dequantize node. We add a pass to retreive the quantization info from the
// quantize node just before the dequantize node and save them on the dequantize
// node.
void SaveDequantInformation(std::shared_ptr<Graph>& graph);

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch