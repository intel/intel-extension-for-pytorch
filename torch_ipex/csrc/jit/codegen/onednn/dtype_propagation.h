#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

enum OutputDtype : int64_t {
    undef=0, // cannot infer OutputDtype of the partition,
             // will set it using the InputDtype of this partition
    uint8, // u8
    int8, // s8
    fp32,
};

void PropagateDtype(const std::shared_ptr<Graph>& graph);

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch