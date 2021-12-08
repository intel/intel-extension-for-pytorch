#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <string>
#include "csrc/jit/cpu/passes/graph_rewrite.h"

namespace torch {
namespace jit {

struct FusionInfo {
  std::string quantized_op_name;
  std::string pattern;
  std::string replacement;
  std::vector<MatchFilter> filters = {};
};

namespace {

std::string getArgList(std::vector<std::string> extra_args) {
  return std::accumulate(
      extra_args.begin(),
      extra_args.end(),
      std::string(),
      [](std::string acc, const std::string& arg) { return acc + ", " + arg; });
}

FusionInfo getIpexFusionInfo(
    const std::string& fp_op_name,
    const std::string& q_op_name,
    const std::vector<std::string>& fp_extra_args,
    const std::vector<std::string>& q_extra_args) {
  const auto& fp_extra_arg_list = getArgList(fp_extra_args);
  const auto& q_extra_arg_list = getArgList(q_extra_args);

  std::string op_pattern = "graph(%a_quant" + fp_extra_arg_list +
      ", %r_scale, %r_zero_point, %r_dtype):" + R"(
          %a_dequant = aten::dequantize(%a_quant)
          %r = )" +
      fp_op_name + "(" + "%a_dequant" + fp_extra_arg_list + ")" + R"(
          %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
          return (%r_quant) )";

  std::string aten_op_pattern = "graph(%a_quant" + fp_extra_arg_list +
      ", %r_scale, %r_zero_point, %r_dtype):" + R"(
      %r_quant = )" +
      q_op_name + "(%a_quant" + q_extra_arg_list + ")" + R"(
      return (%r_quant) )";

  return {q_op_name, op_pattern, aten_op_pattern};
}

} // namespace

void IpexQuantFusion(std::shared_ptr<Graph>& graph) {
  std::vector<FusionInfo> patterns;
  auto adaptive_avg_pool2d_patten = getIpexFusionInfo(
      "aten::adaptive_avg_pool2d",
      "aten::adaptive_avg_pool2d",
      {"%output_size"},
      {"%output_size"});
  auto flatten_patten = getIpexFusionInfo(
      "aten::flatten",
      "aten::flatten",
      {"%start_dim, %end_dim"},
      {"%start_dim, %end_dim"});
  patterns.emplace_back(adaptive_avg_pool2d_patten);
  patterns.emplace_back(flatten_patten);
  for (const auto& info : patterns) {
    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(info.pattern, info.replacement);
    rewriter.runOnGraph(graph, info.filters);
  }
  graph_rewrite::replaceEmbeddingBagWithQEmbeddingBag(graph);
  graph_rewrite::replaceInteractionWithQInteraction(graph);
}

} // namespace jit
} // namespace torch
