#include "csrc/jit/cpu/passes/utils.h"

namespace torch {
namespace jit {
namespace graph_rewrite {
namespace utils {

const std::map<std::string, PostOp>& supported_unary_post_op_fusion_set() {
  // The key of the map is the aten op name to be fused with conv/linear.
  // The value of the map is a struct containing the ideep post op attr key name
  // and the filter of the graph rewriter if any. Example: for an OP aten::xxx,
  // when fusing with conv/linear, the ideep attr is ideep::attr_t::fuse_yyy(),
  // and no filter is required during graph rewrite, thus we need to add
  // {"aten::xxx", {"yyy"}} into the below map table.
  static const std::map<std::string, PostOp> fusion_attr_map{
      {"aten::relu_", {"relu"}},
      {"aten::relu", {"relu"}},
      {"aten::sigmoid", {"sigmoid"}},
      {"aten::sigmoid_", {"sigmoid"}},
      {"aten::silu", {"swish"}},
      {"aten::silu_", {"swish"}},
      {"aten::tanh", {"tanh"}},
      {"aten::tanh_", {"tanh"}},
      {"aten::mish", {"mish"}},
      {"aten::mish_", {"mish"}},
      {"aten::abs", {"abs"}},
      {"aten::abs_", {"abs"}},
      {"aten::exp", {"exp"}},
      {"aten::exp_", {"exp"}},
      {"aten::hardswish", {"hardswish"}},
      {"aten::hardswish_", {"hardswish"}},
      {"aten::square", {"square"}},
      {"aten::square_", {"square"}},
      {"aten::log", {"log"}},
      {"aten::log_", {"log"}},
      {"aten::round", {"round"}},
      {"aten::round_", {"round"}},
      {"aten::sqrt", {"sqrt"}},
      {"aten::sqrt_", {"sqrt"}},
  };
  return fusion_attr_map;
}

// Check if the memory format of the tensor is ChannelsLast(3d)
bool is_channelslast(c10::TensorType tensor) {
  TORCH_CHECK(tensor.dim().has_value());
  int64_t dim = tensor.dim().value();
  std::vector<int64_t> sizes(dim);
  std::vector<int64_t> strides(dim);
  for (int64_t i = 0; i < dim; ++i) {
    TORCH_CHECK(
        tensor.sizes()[i].has_value() && tensor.strides()[i].has_value());
    sizes[i] = tensor.sizes()[i].value();
    strides[i] = tensor.strides()[i].value();
  }
  return (
      c10::is_channels_last_strides_2d(sizes, strides) ||
      c10::is_channels_last_strides_3d(sizes, strides));
}

// Check if the memory format of the tensor is Contiguous
bool is_contiguous(c10::TensorTypePtr tensor) {
  auto tensor_contiguous = tensor->contiguous();
  bool is_contiguous = tensor_contiguous->strides() == tensor->strides();
  return is_contiguous;
}

} // namespace utils
} // namespace graph_rewrite
} // namespace jit
} // namespace torch
