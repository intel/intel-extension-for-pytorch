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
  return calc_values;
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
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %alpha:float, %scale:int, %input_scale:int):
        %r = ipex::conv2d_elu(%a, %w, %b, %stride, %padding, %dilation, %groups, %alpha, %scale, %input_scale)
        return (%r) )";

  std::string conv2d_elu_inplace = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %alpha:float, %scale:int, %input_scale:int):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        %s = aten::elu_(%r, %alpha, %scale, %input_scale)
        return (%s) )";

  std::string conv2d_elu_outplace = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[], %groups:int, %alpha:float, %scale:int, %input_scale:int):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        %s = aten::elu(%r, %alpha, %scale, %input_scale)
        return (%s) )";

  auto filter_conv2d_elu = [] (
      const Match& match,
      const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto scale_value = getIValue("scale", match_vmap, vmap).value();
    auto input_scale_value = getIValue("input_scale", match_vmap, vmap).value();
    bool no_scale = scale_value.isDouble() ? (scale_value.toDouble() == 1.0) : (scale_value.toInt() == 1);
    bool no_input_scale = input_scale_value.isDouble() ? (input_scale_value.toDouble() == 1.0) : (input_scale_value.toInt() == 1);
    return no_scale && no_input_scale;
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
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::_convolution(%a, %w, %b, %stride, %padding, %dilation,
            %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled)
        return (%r) )";

  std::string conv2d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::conv2d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv1d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
        %r = aten::conv1d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv3d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool):
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

} // namespace graph_rewrite
} // namespace jit
} // namespace torch

