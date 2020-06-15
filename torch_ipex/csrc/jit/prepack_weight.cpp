#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <ATen/core/functional.h> 

#include <stack>

#include "cpu/Prepack.h"

namespace torch {
namespace jit {

namespace {

struct ConvParameters {
  at::Tensor conv_w;
  std::vector<int64_t> conv_s;
  std::vector<int64_t> conv_p;
  std::vector<int64_t> conv_d;
  int64_t conv_g;
};

static inline bool hastensor(Module& m, const char* name) {
  return m.hasattr(name) && m.attr(name).isTensor();
}

static inline bool hastuple(Module& m, const char* name) {
  return m.hasattr(name) && m.attr(name).isTuple();
}

static inline bool hasint(Module& m, const char* name) {
  return m.hasattr(name) && m.attr(name).isInt();
}

static inline std::vector<int64_t> tupleToIntList(const IValue& v) {
  return fmap(v.toTuple()->elements(), [](const IValue& v) -> int64_t {
    return v.toInt();
  });
}

class PrepackConvWeightHelper {
  public:
    void analyze(Module& module);
    void transform();
  private:
    bool tryExtractingConvParameters(
      Module& conv,
      ConvParameters& r);
    
    at::Tensor computeUpdatedConvWeight(const ConvParameters& p);
    std::unordered_map<ModulePtr, at::Tensor> conv_module_and_params_;
    std::unordered_map<Graph*, std::vector<std::string>> conv_names_;
};

bool PrepackConvWeightHelper::tryExtractingConvParameters(
    Module& conv,
    ConvParameters& r) {
  if (!hastensor(conv, "weight") || !hastuple(conv, "stride") ||
      !hastuple(conv, "padding") || !hastuple(conv, "dilation") ||
      !hasint(conv, "groups")) {
    return false;
  }
  r.conv_w = conv.attr("weight").toTensor();
  r.conv_s = tupleToIntList(conv.attr("stride"));
  r.conv_p = tupleToIntList(conv.attr("padding"));
  r.conv_d = tupleToIntList(conv.attr("dilation"));
  r.conv_g = conv.attr("groups").toInt();
  return true;
}

at::Tensor PrepackConvWeightHelper::computeUpdatedConvWeight(const ConvParameters& p) {
  at::Tensor new_w= torch_ipex::AtenIpexJITPrepack::prepack_conv_weight(
      p.conv_w, p.conv_p, p.conv_s, p.conv_d, p.conv_g);
  return new_w;
}

void PrepackConvWeightHelper::analyze(Module& module) {
  std::string conv_R = R"(
graph(%self, %x):
    %conv_submodule = match::module[name="Conv2d"](%self)
    %conv_out = prim::CallMethod[name="forward"](%conv_submodule, %x)
    return (%conv_out))";

  Graph pattern_graph;
  std::unordered_map<std::string, Value*> vmap;
  parseIR(conv_R, &pattern_graph, vmap);

  Value* pattern_conv_out = vmap.at("conv_out");
  Value* pattern_conv_submodule = vmap.at("conv_submodule");
  Node* pattern_conv = pattern_conv_out->node();

  std::stack<Module> worklist({module});
  while (!worklist.empty()) {
    Module current = worklist.top();
    worklist.pop();

    for (const Module& submodule : current.children()) {
      worklist.push(submodule);
    }

    for (auto& method : current.get_methods()) {
      GRAPH_DUMP(
          current.type()->name()->name() + "::" + method.name() +
          "() before Conv2d folding", method.graph());
      const auto& matches = findPatternMatches(pattern_graph, *method.graph());
      GRAPH_DEBUG("number of Conv2d matches: ", matches.size());
      Graph* g = method.graph().get();
      if (!conv_names_.count(g)) {
        // This is to make sure we don't visit one graph multiple times
        conv_names_[g] = {};
        for (const Match& match : matches) {
          GRAPH_DEBUG("Checking next match...");
          Node* matched_conv = match.nodes_map.at(pattern_conv);
          Node* matched_conv_submodule = match.values_map.at(pattern_conv_submodule)->node();

          TORCH_INTERNAL_ASSERT(matched_conv_submodule->kind() == prim::GetAttr);
          const auto& conv_module_name = matched_conv_submodule->s(Symbol::attr("name"));
          Module conv_submodule = current.attr(conv_module_name).toModule();

          ConvParameters params;
          if (!tryExtractingConvParameters(conv_submodule, params)) {
            GRAPH_DEBUG("Conv module didn't have all required parameters or attributes...");
            continue;
          }
          conv_names_[g].push_back(conv_module_name); 
        }
      }

      for (const auto& conv : conv_names_.at(g)) {
        Module conv_submodule = current.attr(conv).toModule();
        ConvParameters params;
        TORCH_INTERNAL_ASSERT(tryExtractingConvParameters(conv_submodule, params));
        auto new_w = computeUpdatedConvWeight(params);
        conv_module_and_params_[conv_submodule._ivalue()] = new_w;
      } // conv module
    }
  }
}

void PrepackConvWeightHelper::transform() {
  for (const auto& item : conv_module_and_params_) {
    Module conv(item.first);
    auto w = item.second;
    conv.setattr("weight", w);
  }
}

} // namespace 

Module prepack_conv_weight(const Module& module) {
  PrepackConvWeightHelper h;
  Module m = module.clone();
  h.analyze(m);
  h.transform();
  return m;
}

} // namespace jit

} // namespace torch
