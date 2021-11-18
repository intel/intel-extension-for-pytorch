#include "jit/codegen/onednn/utils.h"

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {
namespace utils {

bool isViewOp(Node* n) {
  switch (n->kind()) {
    case aten::view:
    case aten::permute:
    case aten::transpose:
      return true;
    default:
      return false;
  }
}

bool isEltwiseOp(Node* n) {
  if (n->kind() == Symbol::aten("relu") ||
      n->kind() == Symbol::aten("sigmoid") ||
      n->kind() == Symbol::aten("quantize_per_tensor") ||
      n->kind() == Symbol::aten("quantize_per_channel") ||
      n->kind() == aten::to) {
    return true;
  } else {
    return false;
  }
}

} // namespace utils
} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch