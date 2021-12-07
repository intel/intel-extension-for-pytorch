#include "fusion_group_name.h"

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

const std::string& LlgaFusionGroupName() {
  static const std::string _LlgaFusionGroupName = "ipex::LlgaFusionGroup";
  return _LlgaFusionGroupName;
}

const std::string& LlgaGuardName() {
  static const std::string LlgaGuardName = "ipex::LlgaFusionGuard";
  return LlgaGuardName;
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch