#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <string>
#include "passes/graph_rewrite.h"

namespace torch_ipex {
namespace jit {

struct FusionInfo {
  std::string quantized_op_name;
  std::string pattern;
  std::string replacement;
  std::vector<torch::jit::MatchFilter> filters = {};
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

auto pad_filter =
    [](const torch::jit::Match& match,
       const std::unordered_map<std::string, torch::jit::Value*>& vmap) {
      auto padding_mod = match.values_map.at(vmap.at("padding_mod"));
      if (padding_mod->node()->kind() == torch::jit::prim::Constant) {
        auto padding_mod_value = torch::jit::toIValue(padding_mod).value();
        if (padding_mod_value == "reflect" ||
            padding_mod_value == "replicate") {
          return true;
        } else {
          return false;
        }
      }
      return false;
    };

auto pad_circular_filter =
    [](const torch::jit::Match& match,
       const std::unordered_map<std::string, torch::jit::Value*>& vmap) {
      auto padding_mod = match.values_map.at(vmap.at("padding_mod"));
      if (padding_mod->node()->kind() == torch::jit::prim::Constant) {
        auto padding_mod_value = torch::jit::toIValue(padding_mod).value();
        if (padding_mod_value == "circular") {
          return true;
        } else {
          return false;
        }
      }
      return false;
    };

void IpexQuantFusion(std::shared_ptr<torch::jit::Graph>& graph) {
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
  auto pad_patten = getIpexFusionInfo(
      "aten::pad",
      "aten::pad",
      {"%padding, %padding_mod, %padding_value"},
      {"%padding, %padding_mod, %padding_value"});
  pad_patten.filters.push_back(pad_filter);
  auto pad_circurar_patten = getIpexFusionInfo(
      "aten::pad",
      "ipex::qpad_circular",
      {"%padding, %padding_mod, %padding_value"},
      {"%padding"});
  pad_circurar_patten.filters.push_back(pad_circular_filter);

  patterns.emplace_back(adaptive_avg_pool2d_patten);
  patterns.emplace_back(flatten_patten);
  patterns.emplace_back(pad_patten);
  patterns.emplace_back(pad_circurar_patten);

  for (const auto& info : patterns) {
    torch::jit::SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(info.pattern, info.replacement);
    rewriter.runOnGraph(graph, info.filters);
  }
  GRAPH_DUMP(
      "Before replaceEmbeddingBagWithQEmbeddingBag. Beginning of IpexQuantFusion",
      graph);
  graph_rewrite::replaceEmbeddingBagWithQEmbeddingBag(graph);
  GRAPH_DUMP(
      "After replaceEmbeddingBagWithQEmbeddingBag. Before replaceInteractionWithQInteraction",
      graph);
  graph_rewrite::replaceInteractionWithQInteraction(graph);
  GRAPH_DUMP(
      "After replaceInteractionWithQInteraction. Before replaceMergedEmbCatWithQmergedEmbCat",
      graph);
  graph_rewrite::replaceMergedEmbCatWithQmergedEmbCat(graph);
  GRAPH_DUMP(
      "After replaceMergedEmbCatWithQmergedEmbCat. Before preprocessSizeForQLstm",
      graph);
  graph_rewrite::preprocessSizeForQLstm(graph);
  GRAPH_DUMP(
      "After preprocessSizeForQLstm. Before replaceLstmWithQLstm", graph);
  graph_rewrite::replaceLstmWithQLstm(graph);
  GRAPH_DUMP("After replaceLstmWithQLstm. Before replaceAddWithQAdd", graph);
  graph_rewrite::replaceAddWithQAdd(graph);
  GRAPH_DUMP("After replaceAddWithQAdd. End of IpexQuantFusion", graph);
}

} // namespace jit
} // namespace torch_ipex
