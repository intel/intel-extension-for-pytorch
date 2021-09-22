#include <torch/csrc/jit/runtime/profiling_record.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

bool canFuseNode(const Node* node) {
  // TODO: register all canFuseNode of LLGA here
  return node->kind() == Symbol::aten("quantize_per_tensor") ||
      node->kind() == Symbol::aten("quantize_per_channel") ||
      node->kind() == Symbol::aten("dequantize") ||
      node->kind() == aten::_convolution;
}

namespace {
class RegisterInterface {
 public:
  RegisterInterface() {
    RegisterProfilingNode(canFuseNode);
  }
};

static RegisterInterface register_interface_;
} // namespace

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch