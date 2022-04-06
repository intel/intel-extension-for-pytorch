#include "graph_rewrite_utils.h"
#include <torch/csrc/jit/ir/alias_analysis.h>

namespace torch {
namespace jit {
namespace graph_rewrite {

bool hasSideEffectInDefNode(Node* def_node, int position) {
  bool checkresult = false;
  if (def_node->blocks().size() != 0) {
    // if the def node has blocks, check into the blocks
    for (auto sub : def_node->blocks()) {
      checkresult = checkresult ||
          hasSideEffectInBlocks(sub, def_node->outputs()[position]);
    }
  } else {
    if (def_node->hasAttribute(attr::Subgraph)) {
      // if the def node has subgraph, check into the subgraph
      checkresult = hasSideEffectOrAliasInSubgraphs(
          def_node, def_node->outputs()[position]);
    } else {
      checkresult =
          def_node->hasSideEffects() || (def_node->kind() == prim::Param);
    }
  }

  return checkresult;
}

bool hasSideEffectInBlocks(Block* block, Value* v) {
  bool checkresult = false;
  // find the position of target value in its def node from block outputs
  // for example, here find %block_output.1 == (%input.1) or (%input.2)
  // and the posion is 0:
  // %block_output.1 : Tensor = prim::If()
  //     block0():
  //       %input.1 : Tensor = ipex::LlgaFusionGroup
  //       -> (%input.1)
  //     block1():
  //       %input.2 : Tensor = prim::FallbackGraph
  //       -> (%input.2)
  int position = v->offset();
  auto def_node = block->outputs()[position]->node();
  checkresult = hasSideEffectInDefNode(def_node, position);
  return checkresult;
}

bool hasSideEffectOrAliasInSubgraphs(Node* node, Value* v) {
  bool checkresult = false;
  // A LLGAFusionGroup must have its fallbackgraph, we only need to check one of
  // them
  if (node->kind().toQualString() ==
      Symbol::fromQualString("ipex::LlgaFusionGroup").toQualString()) {
    return false;
  }
  // get the subgraph of the def node
  auto subgraph = node->g(attr::Subgraph);

  // find the position of target value in its def node in subgraph
  // for example, here find (%input.1), and the posion is 0:
  // graph(---),
  //    %input.1 : Tensor = Ops
  //    return (%input.1)
  int position = v->offset();
  auto def_node = subgraph->outputs()[position]->node();
  std::unique_ptr<AliasDb> aliasDb_ = std::make_unique<AliasDb>(subgraph);

  checkresult = hasSideEffectInDefNode(def_node, position);

  // for def node in subgraph, has to check its alias too
  bool mayAliasInputs = (def_node->kind() != prim::ListConstruct) &&
      aliasDb_->mayContainAlias(
          def_node->inputs(), def_node->outputs()[position]);

  checkresult = checkresult || mayAliasInputs;
  return checkresult;
}

bool hasSideEffectOrAlias(Value* v, AliasDb* aliasDb) {
  // bail on the input def node with side effects, blocks, or graph / graph
  // inputs
  Node* n = v->node();
  bool unhandled_node = false;
  if (n->blocks().size() != 0) {
    for (int i = 0; i < n->blocks().size(); i++) {
      unhandled_node =
          unhandled_node || hasSideEffectInBlocks(n->blocks()[i], v);
    }
  } else if (n->hasAttribute(attr::Subgraph)) {
    unhandled_node = hasSideEffectOrAliasInSubgraphs(n, v);
  } else {
    unhandled_node = n->hasSideEffects() || (v->node()->kind() == prim::Param);
  }

  // if the output isn't contained or alias by the inputs to its node, it's
  // unique. No need to check for alias if the node is a ListConstruct.
  bool mayAliasInputs = (v->node()->kind() != prim::ListConstruct) &&
      aliasDb->mayContainAlias(v->node()->inputs(), v);
  return unhandled_node || mayAliasInputs || (v->node()->kind() == prim::Param);
}

auto accumu_use_check = [](const Node* add_node, const Value* accumu_value) {
  bool accumu_same_used = false;
  auto accumu_uses = accumu_value->uses();
  std::for_each(accumu_uses.begin(), accumu_uses.end(), [&](Use& u) {
    // if one user is the after nodes of add. we can't write accumu.
    if (u.user != add_node && !u.user->isBefore(add_node)) {
      accumu_same_used = true;
    }
  });
  return accumu_same_used;
};

MatchFilter fuse_add_filter(std::shared_ptr<Graph>& graph, int accumu_id) {
  // For conv+add, Linear+add fusion path, there have some conditions need to
  // be meet:
  // 1: The conv/linear's input should not be accumu.
  // 2: The accumu should be a tensor and it will not be used by the downstream
  // ops of add. 3: The shapes of add's inputs should be same.
  // 4. For output = Y + alpha*op_output, the alpha should be 1.0.
  // 5. The Accumu should not be an alias of other tensor.

  return [graph, accumu_id](
             const Match& match,
             const std::unordered_map<std::string, Value*>& vmap) {
    auto add_node = match.values_map.at(vmap.at("res"))->node();
    auto add_x = match.values_map.at(vmap.at("input"));
    auto accumu = match.values_map.at(vmap.at("accumu"));
    std::unique_ptr<AliasDb> aliasDb_ = std::make_unique<AliasDb>(graph);
    // For output = input + alpha*op(input_alias), we can't do add fusion add.
    if (aliasDb_.get()->mayContainAlias(accumu, add_x)) {
      return false;
    }
    bool accumu_same_used = accumu_use_check(add_node, accumu);
    // accumu is used by other ops(post ops of add) or it is a constant or is
    // not tensor.
    if (accumu_same_used || accumu->node()->kind() == prim::Constant ||
        !accumu->type()->cast<TensorType>()) {
      return false;
    }
    // Check inputs of add have same shapes.
    auto size1_option = add_node->inputs()
                            .at(0)
                            ->type()
                            ->cast<TensorType>()
                            ->sizes()
                            .concrete_sizes();
    auto size2_option = add_node->inputs()
                            .at(1)
                            ->type()
                            ->cast<TensorType>()
                            ->sizes()
                            .concrete_sizes();
    if (!size1_option.has_value() || !size2_option.has_value()) {
      return false;
    }
    auto size1_vec = size1_option.value();
    auto size2_vec = size2_option.value();
    if (size1_vec.empty() || size2_vec.empty() || size1_vec != size2_vec) {
      return false;
    }
    // output = Y + alpha*op_output, alpha need has value and the value should
    // be 1.0.
    if (!accumu_id && vmap.find("alpha") != vmap.end()) {
      auto alpha = toIValue(match.values_map.at(vmap.at("alpha")));
      if (alpha.has_value() && alpha.value().isDouble()) {
        auto alpha_ = alpha.value().toDouble();
        if (alpha_ != 1.0) {
          return false;
        }
      }
    }
    // For Y = Y + alpha*op_output, don't need do alias check.
    if (!accumu_id && add_node->kind() == aten::add_) {
      return true;
    }
    // Skip if input's def node has side effect or input has alias
    if (hasSideEffectOrAlias(
            add_node->inputs().at(accumu_id), aliasDb_.get())) {
      return false;
    }
    return true;
  };
}

} // namespace graph_rewrite
} // namespace jit
} // namespace torch
