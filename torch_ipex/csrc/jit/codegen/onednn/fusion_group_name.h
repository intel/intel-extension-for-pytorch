#pragma once

#include <string>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// Workaround here. Once the PR of PyTorch LLGA bridge code has been landed
// into the stock PyTorch, we could directly use the Symbol: prim::LlgaFusionGroup
// instead of Symbol::fromQualString(LlgaFusionGroupName())
extern const std::string& LlgaFusionGroupName();

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch