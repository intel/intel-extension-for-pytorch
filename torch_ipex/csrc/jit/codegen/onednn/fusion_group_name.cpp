#include "fusion_group_name.h"

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

const std::string& LlgaFusionGroupName() {
  static const std::string _LlgaFusionGroupName = "ipex::LlgaFusionGroup";
  return _LlgaFusionGroupName;
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch