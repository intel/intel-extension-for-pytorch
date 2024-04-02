#pragma once

#include <Macros.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch_ipex {
namespace jit {
namespace graph_rewrite_helper {

// those code just copy from PyTorch offical and extend
// replaceConvolutionWithAtenConv to handle conv_transpose3d.

torch::jit::Value* getValue(
    const std::string& name,
    const std::unordered_map<const torch::jit::Value*, torch::jit::Value*>&
        match_vmap,
    const std::unordered_map<std::string, torch::jit::Value*>& vmap);
c10::optional<c10::IValue> getIValue(
    const std::string& name,
    const std::unordered_map<const torch::jit::Value*, torch::jit::Value*>&
        match_vmap,
    const std::unordered_map<std::string, torch::jit::Value*>& vmap);
IPEX_API void replaceConvolutionWithAtenConv(
    std::shared_ptr<torch::jit::Graph>& graph);

bool isClampFusable(
    const torch::jit::Match& match,
    const std::unordered_map<std::string, torch::jit::Value*>& vmap);

void insertBias(
    torch::jit::Graph* graph,
    torch::jit::Node* node,
    c10::optional<at::Tensor> bias);

// This struct contains a compiled IR patterns slated for use in the
// findPatternMatches function. The struct encapsulates the common
// information from parseIR that is used in conjunction with the
// pattern matching facility. A const instance of this struct can
// also be stored away to cache the compiled IR pattern and reduce
// runtime cost
struct PatternInfo {
  std::string pattern_string;
  std::unique_ptr<torch::jit::Graph> pattern_graph;
  std::unordered_map<std::string, torch::jit::Value*> vmap;
  std::vector<torch::jit::MatchFilter> filters;

  static PatternInfo parse_from_str(
      std::string pattern_string,
      const std::vector<torch::jit::MatchFilter>& filters = {}) {
    PatternInfo rv{
        std::move(pattern_string),
        std::make_unique<torch::jit::Graph>(),
        decltype(vmap){},
        filters};
    parseIR(rv.pattern_string, rv.pattern_graph.get(), rv.vmap);
    return rv;
  }
};

} // namespace graph_rewrite_helper
} // namespace jit
} // namespace torch_ipex
