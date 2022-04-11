#include "graph_rewrite.h"
#include "graph_rewrite_utils.h"

#include <ATen/code_template.h>

namespace torch {
namespace jit {
namespace graph_rewrite {

using namespace at::jit;

auto ipex_einsum_filter =
    [](const Match& match,
       const std::unordered_map<std::string, Value*>& vmap) {
      const auto& match_vmap = match.values_map;
      auto equation =
          getIValue("equation", match_vmap, vmap).value().toStringView();
      int num_ops = std::count(equation.begin(), equation.end(), ',') + 1;
      if (num_ops != 2)
        return false; // only process the 2 operands
      return true;
    };

void FusedEinsumPost(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter_einsum_binary;
  std::array<std::string, 2> binarys = {"add", "add_"};
  auto aten_einsum_binary = CodeTemplate(R"(
     graph(%equation, %inputs, %add_arg, %alpha):
        %x = aten::einsum(%equation, %inputs)
        %res = aten::${binary}(%x, %add_arg, %alpha)
        return (%res))");
  std::string fused_einsum_binary = R"(
    graph(%equation, %inputs, %add_arg, %alpha):
        %res = ipex::einsum_binary(%equation, %inputs, %add_arg, %alpha)
        return (%res))";

  for (const auto& binary : binarys) {
    TemplateEnv env;
    env.s("binary", binary);
    rewriter_einsum_binary.RegisterRewritePattern(
        aten_einsum_binary.format(env), fused_einsum_binary);
  }
  rewriter_einsum_binary.runOnGraph(graph, ipex_einsum_filter);
}

} // namespace graph_rewrite
} // namespace jit
} // namespace torch
