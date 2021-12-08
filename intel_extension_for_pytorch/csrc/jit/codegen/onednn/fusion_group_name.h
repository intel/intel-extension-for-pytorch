#pragma once

#include <string>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// Workaround here. Once the PR of PyTorch LLGA bridge code has been landed
// into the stock PyTorch, we could directly use the Symbol:
// prim::LlgaFusionGroup and prim::LlgaFusionGuard instead of
// Symbol::fromQualString(LlgaFusionGroupName()) and
// Symbol::fromQualString(LlgaGuardName())
extern const std::string& LlgaFusionGroupName();
extern const std::string& LlgaGuardName();

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch