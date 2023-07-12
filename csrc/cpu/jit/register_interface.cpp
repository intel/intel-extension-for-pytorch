#include <torch/csrc/jit/runtime/profiling_record.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

bool canFuseNode(const Node* node) {
  // Switch does not allow non-constexpr function, to make the Symbol constexpr,
  // we must add them to the list in aten/src/ATen/core/interned_strings.h to
  // explicitly use interned strings as symbols. Use if-else here instead
  return (
      node->kind() == Symbol::aten("quantize_per_tensor") ||
      node->kind() == Symbol::aten("quantize_per_channel") ||
      node->kind() == Symbol::aten("dequantize") ||
      node->kind() == Symbol::aten("_convolution") ||
      node->kind() == Symbol::aten("conv2d") ||
      node->kind() == Symbol::aten("add") ||
      node->kind() == Symbol::aten("add_") ||
      node->kind() == Symbol::aten("div") ||
      node->kind() == Symbol::aten("tanh") ||
      node->kind() == Symbol::aten("relu") ||
      node->kind() == Symbol::aten("elu") ||
      node->kind() == Symbol::aten("sigmoid") ||
      node->kind() == Symbol::aten("gelu") ||
      node->kind() == Symbol::aten("sqrt") ||
      node->kind() == Symbol::aten("abs") ||
      node->kind() == Symbol::aten("square") ||
      node->kind() == Symbol::aten("hardtanh") ||
      node->kind() == Symbol::aten("softmax") ||
      node->kind() == Symbol::aten("max_pool2d") ||
      node->kind() == Symbol::aten("avg_pool2d") ||
      node->kind() == Symbol::aten("matmul") ||
      node->kind() == Symbol::aten("mm") ||
      node->kind() == Symbol::aten("linear") ||
      node->kind() == Symbol::aten("batch_norm") ||
      node->kind() == Symbol::aten("layer_norm") ||
      node->kind() == Symbol::aten("masked_fill") ||
      node->kind() == Symbol::aten("masked_fill_") ||
      node->kind() == Symbol::aten("pad") ||
      node->kind() == Symbol::aten("mul") ||
      node->kind() == Symbol::aten("flatten") ||
      node->kind() ==
          Symbol::fromQualString("torch_ipex::convolution_forward") ||
      node->kind() == Symbol::fromQualString("torch_ipex::ipex_linear") ||
      node->kind() == Symbol::fromQualString("torch_ipex::conv_transpose") ||
      node->kind() == Symbol::fromQualString("torch_ipex::ipex_MKLSGEMM"));
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
