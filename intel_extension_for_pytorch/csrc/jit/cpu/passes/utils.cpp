#include "csrc/jit/cpu/passes/utils.h"
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>

namespace torch {
namespace jit {
namespace graph_rewrite {
namespace utils {

bool aten_elu_no_input_scale(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  const auto& match_vmap = match.values_map;
  auto input_scale_value =
      graph_rewrite_helper::getIValue("input_scale", match_vmap, vmap).value();
  bool no_input_scale = input_scale_value.isDouble()
      ? (input_scale_value.toDouble() == 1.0)
      : (input_scale_value.toInt() == 1);
  return no_input_scale;
}

bool aten_clamp_min_max_not_none(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  const auto& match_vmap = match.values_map;
  auto min_value =
      graph_rewrite_helper::getIValue("min", match_vmap, vmap).value();
  auto max_value =
      graph_rewrite_helper::getIValue("max", match_vmap, vmap).value();
  return !min_value.isNone() && !max_value.isNone();
}

bool aten_pow_exponent_is_scalar(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  const auto& match_vmap = match.values_map;
  auto exponent_value =
      graph_rewrite_helper::getIValue("exponent", match_vmap, vmap).value();
  return exponent_value.isScalar();
}

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

const std::map<std::string, NonUnaryPostOp>&
supported_non_unary_post_op_fusion_set() {
  // Compared with supported_unary_post_op_fusion_set(),
  // for non-unary post OP to be fused with conv/linear,
  // need to provide an extra field op_input_list which is a vector
  // of string containing the inputs to the non-unary post OP (except for the
  // input tensor itself) For example, for an OP:
  //   aten::(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1)
  // the op_input_list should be set to:
  //   std::vector<std::string>({"%alpha", "%scale", "%input_scale"}).
  static const std::map<std::string, NonUnaryPostOp> fusion_attr_map{
      {"aten::leaky_relu",
       {"leaky_relu", std::vector<std::string>({"%alpha"})}},
      {"aten::leaky_relu_",
       {"leaky_relu", std::vector<std::string>({"%alpha"})}},
      {"aten::hardtanh",
       {"hardtanh", std::vector<std::string>({"%min", "%max"})}},
      {"aten::hardtanh_",
       {"hardtanh", std::vector<std::string>({"%min", "%max"})}},
      {"aten::elu",
       {"elu",
        std::vector<std::string>({"%alpha", "%scale", "%input_scale"}),
        {aten_elu_no_input_scale}}},
      {"aten::elu_",
       {"elu",
        std::vector<std::string>({"%alpha", "%scale", "%input_scale"}),
        {aten_elu_no_input_scale}}},
      {"aten::clamp",
       {"hardtanh",
        std::vector<std::string>({"%min", "%max"}),
        {aten_clamp_min_max_not_none}}},
      {"aten::clamp_",
       {"hardtanh",
        std::vector<std::string>({"%min", "%max"}),
        {aten_clamp_min_max_not_none}}},
      {"aten::pow",
       {"pow",
        std::vector<std::string>({"%exponent"}),
        {aten_pow_exponent_is_scalar}}},
      {"aten::pow_",
       {"pow",
        std::vector<std::string>({"%exponent"}),
        {aten_pow_exponent_is_scalar}}},
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
  if (!tensor->sizes().concrete_sizes().has_value()) {
    return false;
  }
  auto tensor_contiguous = tensor->contiguous();
  bool is_contiguous = tensor_contiguous->strides() == tensor->strides();
  return is_contiguous;
}

} // namespace utils
} // namespace graph_rewrite
} // namespace jit
} // namespace torch
