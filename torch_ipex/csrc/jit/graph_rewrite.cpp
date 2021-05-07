#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include "graph_rewrite.h"

namespace torch {
namespace jit {
namespace graph_rewrite {

// those code just copy from PyTorch offical:
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/passes/graph_rewrite_helper.cpp

Value* getValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap) {
  return match_vmap.at(vmap.at(name));
}

c10::optional<IValue> getIValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap) {
  return toIValue(getValue(name, match_vmap, vmap));
}

std::unordered_map<std::string, c10::IValue> getConvParams(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  std::unordered_map<std::string, c10::IValue> calc_values;
  const auto& match_vmap = match.values_map;
  auto transposed_value = getIValue("transposed", match_vmap, vmap).value();
  calc_values["transposed"] = transposed_value;
  auto benchmark_value = getIValue("benchmark", match_vmap, vmap).value();
  calc_values["benchmark"] = benchmark_value;
  auto deterministic_value =
      getIValue("deterministic", match_vmap, vmap).value();
  calc_values["deterministic"] = deterministic_value;
  auto cudnn_enabled_value =
      getIValue("cudnn_enabled", match_vmap, vmap).value();
  calc_values["cudnn_enabled"] = cudnn_enabled_value;
  auto output_padding_value =
      getIValue("output_padding", match_vmap, vmap).value();
  calc_values["output_padding"] = output_padding_value;
  auto stride_value = getIValue("stride", match_vmap, vmap).value();
  calc_values["stride"] = stride_value;
  auto padding_value = getIValue("padding", match_vmap, vmap).value();
  calc_values["padding"] = padding_value;
  auto dilation_value = getIValue("dilation", match_vmap, vmap).value();
  calc_values["dilation"] = dilation_value;
  auto allow_tf32_value = getIValue("allow_tf32", match_vmap, vmap).value();
  calc_values["allow_tf32_value"] = allow_tf32_value;
  return calc_values;
}

void FuseShuffle(std::shared_ptr<Graph>& graph) {
  std::string shuffle = R"(
      graph(%input, %view_shape:int[], %trans_dim0:int, %trans_dim1:int, %mem_format:int, %flattern_shape:int[]):
        %r1 = aten::view(%input, %view_shape)
        %r2 = aten::transpose(%r1, %trans_dim0, %trans_dim1)
        %r3 = aten::contiguous(%r2, %mem_format)
        %r4 = aten::view(%r3, %flattern_shape)
        return (%r4) )";

  std::string shuffle_2d_fusion = R"(
      graph(%input, %view_shape:int[], %trans_dim0:int, %trans_dim1:int, %mem_format:int, %flattern_shape:int[]):
        %r = ipex::shuffle_2d(%input, %view_shape, %trans_dim0, %trans_dim1)
        return (%r) )";

  auto filter_shuffle_2d_fusion = [] (
      const Match& match,
      const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto input_ = getIValue("input", match_vmap, vmap).value();
    if (!(input_.isTensor())) {
      return false;
    }
    auto view_shape_ = getIValue("view_shape", match_vmap, vmap).value();
    if (!(view_shape_.isIntList())) {
      return false;
    }
    auto trans_dim0_ = getIValue("trans_dim0", match_vmap, vmap).value();
    if (!(trans_dim0_.isInt())) {
      return false;
    }
    auto trans_dim1_ = getIValue("trans_dim1", match_vmap, vmap).value();
    if (!(trans_dim1_.isInt())) {
      return false;
    }
    auto flattern_shape_ = getIValue("flattern_shape", match_vmap, vmap).value();
    if (!(flattern_shape_.isInt())) {
      return false;
    }

    auto trans_dim0_val = trans_dim0_.toInt();
    auto trans_dim1_val = trans_dim1_.toInt();
    auto dim0_val = trans_dim0_val < trans_dim1_val ? trans_dim0_val : trans_dim1_val;
    auto dim1_val = trans_dim0_val > trans_dim1_val ? trans_dim0_val : trans_dim1_val;
    // If the tranpose if not for groups. ex. [n, c1, c2, h, w] => [n, c2, c1, h, w]
    if ((dim1_val - dim0_val) != 1) {
      return false;
    }

    auto input_val = input_.toTensor();
    auto view_shape_val = view_shape_.toIntVector();
    auto flattern_shape_val = flattern_shape_.toIntVector();
    // ex. [n, c, h, w] => [n, groups, c // groups, h, w]
    if ((input_val.ndimension() - view_shape_val.size()) != -1) {
      return false;
    }

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim0_val >= 0);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim1_val >= 0);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim0_val + 1 < input_val.ndimension());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim1_val + 1 < input_val.ndimension());
    if (view_shape_val[dim0_val] * view_shape_val[dim1_val] != input_val.size(dim0_val)) {
      return false;
    }

    if (flattern_shape_val.size() != input_val.ndimension()) {
      return false;
    }

    for (int i = 0; i < flattern_shape_val.size(); i++) {
      if (flattern_shape_val[i] != input_val.size(i)) {
        // [n, c, h, w] => view [n, groups, c // groups, h, w] => tranpose [n, c // groups, groups, h, w]
        // => view [n, -1, h, w]
        //    or
        //    view [n, c, h, w]
        if ((flattern_shape_val[i] != -1) || (i != dim0_val)) {
          return false;
        }
      }
    }

    return true;
  };

  SubgraphRewriter rewriter_shuffle_2d;
  rewriter_shuffle_2d.RegisterRewritePattern(
    shuffle,
    shuffle_2d_fusion);
  rewriter_shuffle_2d.runOnGraph(graph);
}

void FuseConvolutionWithEltwise(std::shared_ptr<Graph>& graph) {
  std::string conv2d_swish_fusion = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = ipex::conv2d_swish(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv2d_sigmoid_mul_outplace = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        %s = aten::sigmoid(%r)
        %t = aten::mul(%r, %s)
        return (%t) )";

  std::string conv2d_sigmoid_mul_inplace = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        %s = aten::sigmoid(%r)
        %t = aten::mul_(%r, %s)
        return (%t) )";

  std::string conv2d_sigmoid_fusion = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = ipex::conv2d_sigmoid(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv2d_sigmoid_outplace = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        %s = aten::sigmoid(%r)
        return (%s) )";

  std::string conv2d_sigmoid_inplace = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        %s = aten::sigmoid_(%r)
        return (%s) )";

  std::string conv2d_hardtanh_fusion = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %min:float, %max:float):
        %r = ipex::conv2d_clamp(%a, %w, %b, %stride, %padding, %dilation, %groups, %min, %max)
        return (%r) )";

  std::string conv2d_hardtanh_inplace = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %min, %max):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        %s = aten::hardtanh_(%r, %min, %max)
        return (%s) )";

  std::string conv2d_hardtanh_outplace = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %min, %max):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        %s = aten::hardtanh(%r, %min, %max)
        return (%s) )";

  std::string conv2d_elu_fusion = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %alpha:float, %scale, %input_scale):
        %r = ipex::conv2d_elu(%a, %w, %b, %stride, %padding, %dilation, %groups, %alpha, %scale, %input_scale)
        return (%r) )";

  std::string conv2d_elu_inplace = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %alpha:float, %scale, %input_scale):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        %s = aten::elu_(%r, %alpha, %scale, %input_scale)
        return (%s) )";

  std::string conv2d_elu_outplace = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %alpha:float, %scale, %input_scale):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        %s = aten::elu(%r, %alpha, %scale, %input_scale)
        return (%s) )";

  auto filter_conv2d_elu = [] (
      const Match& match,
      const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto input_scale_value = getIValue("input_scale", match_vmap, vmap).value();
    bool no_input_scale = input_scale_value.isDouble() ? (input_scale_value.toDouble() == 1.0) : (input_scale_value.toInt() == 1);
    return no_input_scale;
  };

  // Fuse conv2d + swish
  SubgraphRewriter rewriter_conv_swish_outplace;
  rewriter_conv_swish_outplace.RegisterRewritePattern(
    conv2d_sigmoid_mul_outplace,
    conv2d_swish_fusion);
  rewriter_conv_swish_outplace.runOnGraph(graph);
  SubgraphRewriter rewriter_conv_swish_inplace;
  rewriter_conv_swish_inplace.RegisterRewritePattern(
    conv2d_sigmoid_mul_inplace,
    conv2d_swish_fusion);
  rewriter_conv_swish_inplace.runOnGraph(graph);

  // Fuse conv2d + sigmoid
  SubgraphRewriter rewriter_conv_sigmoid_outplace;
  rewriter_conv_sigmoid_outplace.RegisterRewritePattern(
    conv2d_sigmoid_outplace,
    conv2d_sigmoid_fusion);
  rewriter_conv_sigmoid_outplace.runOnGraph(graph);
  SubgraphRewriter rewriter_conv_sigmoid_inplace;
  rewriter_conv_sigmoid_inplace.RegisterRewritePattern(
    conv2d_sigmoid_inplace,
    conv2d_sigmoid_fusion);
  rewriter_conv_sigmoid_inplace.runOnGraph(graph);

  // Fuse conv2d + hardtanh
  SubgraphRewriter rewriter_conv_hardtanh_outplace;
  rewriter_conv_hardtanh_outplace.RegisterRewritePattern(
    conv2d_hardtanh_outplace,
    conv2d_hardtanh_fusion);
  rewriter_conv_hardtanh_outplace.runOnGraph(graph);
  SubgraphRewriter rewriter_conv_hardtanh_inplace;
  rewriter_conv_hardtanh_inplace.RegisterRewritePattern(
    conv2d_hardtanh_inplace,
    conv2d_hardtanh_fusion);
  rewriter_conv_hardtanh_inplace.runOnGraph(graph);

  // Fuse conv2d + elu
  SubgraphRewriter rewriter_conv_elu_outplace;
  rewriter_conv_elu_outplace.RegisterRewritePattern(
    conv2d_elu_outplace,
    conv2d_elu_fusion);
  rewriter_conv_elu_outplace.runOnGraph(graph, filter_conv2d_elu);
  SubgraphRewriter rewriter_conv_elu_inplace;
  rewriter_conv_elu_inplace.RegisterRewritePattern(
    conv2d_elu_inplace,
    conv2d_elu_fusion);
  rewriter_conv_elu_inplace.runOnGraph(graph, filter_conv2d_elu);
}

void replaceConvolutionWithAtenConv(std::shared_ptr<Graph>& graph) {
  ConstantPropagation(graph);
  std::string convolution = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::_convolution(%a, %w, %b, %stride, %padding, %dilation,
            %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled, %allow_tf32)
        return (%r) )";

  std::string conv2d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv1d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv1d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv3d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv3d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  // Filter the unsupported case
  auto filter_conv1d = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    auto calc_value_map = getConvParams(match, vmap);
    if (calc_value_map["output_padding"].toIntList().size() != 1 ||
        calc_value_map["stride"].toIntList().size() != 1 ||
        calc_value_map["padding"].toIntList().size() != 1 ||
        calc_value_map["dilation"].toIntList().size() != 1) {
      return false;
    }
    return !calc_value_map["transposed"].toBool() &&
        !calc_value_map["benchmark"].toBool() &&
        !calc_value_map["deterministic"].toBool() &&
        calc_value_map["cudnn_enabled"].toBool() &&
        (calc_value_map["output_padding"].toIntList()[0] == 0);
  };

  auto filter_conv2d = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    auto calc_value_map = getConvParams(match, vmap);
    if (calc_value_map["output_padding"].toIntList().size() != 2 ||
        calc_value_map["stride"].toIntList().size() != 2 ||
        calc_value_map["padding"].toIntList().size() != 2 ||
        calc_value_map["dilation"].toIntList().size() != 2) {
      return false;
    }
    auto b1 = calc_value_map["transposed"].toBool();
    auto b2 = calc_value_map["benchmark"].toBool();
    auto b3 = calc_value_map["deterministic"].toBool();
    auto b4 = calc_value_map["cudnn_enabled"].toBool();
    auto b5 = (calc_value_map["output_padding"].toIntList()[0] == 0);
    auto b6 = (calc_value_map["output_padding"].toIntList()[1] == 0);
    return !b1 && !b2 && !b3 && b4 && b5 && b6;
  };

  auto filter_conv3d = [](const Match& match,
                          const std::unordered_map<std::string, Value*>& vmap) {
    auto calc_value_map = getConvParams(match, vmap);
    if (calc_value_map["output_padding"].toIntList().size() != 3 ||
        calc_value_map["stride"].toIntList().size() != 3 ||
        calc_value_map["padding"].toIntList().size() != 3 ||
        calc_value_map["dilation"].toIntList().size() != 3) {
      return false;
    }
    return !calc_value_map["transposed"].toBool() &&
        !calc_value_map["benchmark"].toBool() &&
        !calc_value_map["deterministic"].toBool() &&
        calc_value_map["cudnn_enabled"].toBool() &&
        (calc_value_map["output_padding"].toIntList()[0] == 0) &&
        (calc_value_map["output_padding"].toIntList()[1] == 0) &&
        (calc_value_map["output_padding"].toIntList()[2] == 0);
  };

  SubgraphRewriter rewriter_conv1d;
  rewriter_conv1d.RegisterRewritePattern(convolution, conv1d);
  rewriter_conv1d.runOnGraph(graph, filter_conv1d);
  SubgraphRewriter rewriter_conv2d;
  rewriter_conv2d.RegisterRewritePattern(convolution, conv2d);
  rewriter_conv2d.runOnGraph(graph, filter_conv2d);
  SubgraphRewriter rewriter_conv3d;
  rewriter_conv3d.RegisterRewritePattern(convolution, conv3d);
  rewriter_conv3d.runOnGraph(graph, filter_conv3d);
}

void replaceAtenConvolutionWithIpexConv(std::shared_ptr<Graph>& graph) {
  std::string conv2d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";
  std::string ipex_conv2d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = ipex::conv2d_base(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";
  SubgraphRewriter rewriter_conv2d;
  rewriter_conv2d.RegisterRewritePattern(conv2d, ipex_conv2d);
  rewriter_conv2d.runOnGraph(graph);
 }

} // namespace graph_rewrite
} // namespace jit
} // namespace torch

