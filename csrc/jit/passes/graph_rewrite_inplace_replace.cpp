#include "graph_rewrite_inplace_replace.h"
#include "codegen/onednn/remove_mutation.h"

namespace torch_ipex {
namespace jit {
namespace graph_rewrite {

using namespace torch::jit;

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
  // A LLGAFusionGroup or TensorExprGroup must have its fallbackgraph, we only
  // need to check one of them
  if (node->kind().toQualString() ==
      Symbol::fromQualString("ipex::LlgaFusionGroup").toQualString()) {
    return false;
  }
  if (node->kind().toQualString() ==
      Symbol::fromQualString("prim::TensorExprGroup").toQualString()) {
    return false;
  }

  // get the subgraph of the def node
  auto subgraph = node->g(attr::Subgraph);

  // find the position of target value in its def node in subgraph
  // for example, here find (%input.1), and the posion is 0:
  // graph(---),
  //    %input.1 : Tensor = Ops
  //    %input.2 : Tensor = Ops
  //    return (%input.1, %input.2)

  // position_in_subgraph is graph returned position, e.g, for %input.1 is 0,
  // for %input.2 is 1
  int position_in_subgraph = v->offset();
  auto def_node = subgraph->outputs()[position_in_subgraph]->node();
  // position_in_def_node is def node position, e.g, for %input.1 or %input.2 is
  // 0
  int position_in_def_node =
      subgraph->outputs()[position_in_subgraph]->offset();

  checkresult = hasSideEffectInDefNode(def_node, position_in_def_node);

  // for def node in subgraph, has to check its alias too
  // if the output isn't contained or alias by the inputs to its node, it's
  // unique. No need to check for alias if the node is a ListConstruct.
  std::unique_ptr<AliasDb> aliasDb_ = std::make_unique<AliasDb>(subgraph);
  bool mayAliasInputs = (def_node->kind() != prim::ListConstruct) &&
      aliasDb_->mayContainAlias(
          def_node->inputs(), def_node->outputs()[position_in_def_node]);
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

void replaceAtenOpsWithIpexInplaceOps(std::shared_ptr<Graph>& graph) {
  std::string aten_softmax = R"(
      graph(%a, %dim:int, %half_to_float:bool):
        %r = aten::softmax(%a, %dim, %half_to_float)
        return (%r) )";
  std::string ipex_softmax = R"(
      graph(%a, %dim:int, %half_to_float:bool):
        %r = ipex::softmax(%a, %dim, %half_to_float)
        return (%r) )";
  std::string ipex_softmax_ = R"(
      graph(%a, %dim:int, %half_to_float:bool):
        %r = ipex::softmax_(%a, %dim, %half_to_float)
        return (%r) )";

  // Filter the unsupported case for inplace softmax
  // for contiguous input:
  // replace aten::softmax to ipex::softmax_ during jit pass
  // there is better performance for ipex::softmax_ with oneDNN than
  // aten::softmax
  // for non-contiguous input:
  // (1) oneDNN will use ref path which is not optimized as expected
  // (2) if do contiguous copy then go into oneDNN optimized path, the
  // copy overhead is unneglectable
  // (3) so here will not replace aten::softmax to avoid unexpected regression
  auto filter_inplace_for_softmax =
      [graph](
          const Match& match,
          const std::unordered_map<std::string, Value*>& vmap) {
        Node* node = match.anchor;
        std::unique_ptr<AliasDb> aliasDb_ = std::make_unique<AliasDb>(graph);

        // check if the input is contiguous, and skip if it is not
        auto input_value = node->input(0)->type()->cast<TensorType>();
        if (!utils::is_contiguous(input_value)) {
          return false;
        }

        // Skip if input has more than one use
        if (node->input(0)->uses().size() > 1) {
          return false;
        }

        // Skip if input's def node has side effect or input has alias
        if (hasSideEffectOrAlias(node->inputs().at(0), aliasDb_.get())) {
          return false;
        }
        return true;
      };

  SubgraphRewriter rewriter_aten_inplace;
  SubgraphRewriter rewriter_ipex_inplace;
  rewriter_aten_inplace.RegisterRewritePattern(aten_softmax, ipex_softmax_);
  rewriter_ipex_inplace.RegisterRewritePattern(ipex_softmax, ipex_softmax_);
  rewriter_aten_inplace.runOnGraph(graph, filter_inplace_for_softmax);
  rewriter_ipex_inplace.runOnGraph(graph, filter_inplace_for_softmax);
}

// based on the aten inplace op list:
// {PyTorch Repo}:torch/csrc/jit/passes/restore_mutation.h#L14-L31
std::unordered_map<std::string, int> aten_ops_args_mapping = {
    {"aten::silu", 0},
    {"aten::sigmoid", 0},
    {"aten::tanh", 0},
    {"aten::hardsigmoid", 0},
    {"aten::hardswish", 0},
    {"aten::relu6", 0},
    {"aten::relu", 0},
    {"aten::selu", 0},
    {"aten::celu", 1},
    {"aten::leaky_relu", 1},
    {"aten::hardtanh", 2},
    {"aten::elu", 3},
    {"aten::rrelu", 4}};

void replaceOpsWithAtenInplaceOps(std::shared_ptr<Graph>& graph) {
  std::string input_no_args = R"(
      graph(%input):)";
  std::string input_with_one_args = R"(
      graph(%input, %arg1):)";
  std::string input_with_two_args = R"(
      graph(%input, %arg1, %arg2):)";
  std::string input_with_three_args = R"(
      graph(%input, %arg1, %arg2, %arg3):)";
  std::string input_with_four_args = R"(
      graph(%input, %arg1, %arg2, %arg3, %arg4):)";

  std::string set_result = R"(
       %_result = )";
  std::string ops_no_args = R"((%input) )";
  std::string ops_with_one_args = R"((%input, %arg1) )";
  std::string ops_with_two_args = R"((%input, %arg1, %arg2) )";
  std::string ops_with_three_args = R"((%input, %arg1, %arg2, %arg3) )";
  std::string ops_with_four_args = R"((%input, %arg1, %arg2, %arg3, %arg4) )";
  std::string set_return = R"(
       return (%_result) )";

  // Filter the unsupported cases
  auto filter_inplace =
      [graph](
          const Match& match,
          const std::unordered_map<std::string, Value*>& vmap) {
        Node* node = match.anchor;
        std::unique_ptr<AliasDb> aliasDb_ = std::make_unique<AliasDb>(graph);
        Value* input = node->inputs().at(0);
        Value* output = node->outputs().at(0);
        auto inputDtype = input->type()->expect<TensorType>()->scalarType();
        auto outputDtype = output->type()->expect<TensorType>()->scalarType();

        // If type promotion (from the op list that needs to do check) is
        // allowed, then perform dtype check
        bool check_dtype =
            activation_type_promotion_mapping.find(node->kind()) !=
                activation_type_promotion_mapping.end()
            ? activation_type_promotion_mapping.at(node->kind())
            : false;
        if (check_dtype &&
            (!inputDtype || !outputDtype ||
             inputDtype.value() != outputDtype.value())) {
          return false;
        }

        // Skip if input has more than one use
        if (node->input(0)->uses().size() > 1) {
          return false;
        }

        // Skip if input's def node has side effect or input has alias
        if (hasSideEffectOrAlias(node->inputs().at(0), aliasDb_.get())) {
          return false;
        }
        return true;
      };

  SubgraphRewriter rewriter_aten_inplace;
  for (auto it = aten_ops_args_mapping.begin();
       it != aten_ops_args_mapping.end();
       it++) {
    std::string name = it->first;
    int args_num = it->second;
    std::string match_pattern = "";
    std::string inplace_pattern = "";
    if (args_num == 0) {
      match_pattern =
          input_no_args + set_result + name + ops_no_args + set_return;
      inplace_pattern =
          input_no_args + set_result + name + R"(_)" + ops_no_args + set_return;
    } else if (args_num == 1) {
      match_pattern = input_with_one_args + set_result + name +
          ops_with_one_args + set_return;
      inplace_pattern = input_with_one_args + set_result + name + R"(_)" +
          ops_with_one_args + set_return;
    } else if (args_num == 2) {
      match_pattern = input_with_two_args + set_result + name +
          ops_with_two_args + set_return;
      inplace_pattern = input_with_two_args + set_result + name + R"(_)" +
          ops_with_two_args + set_return;
    } else if (args_num == 3) {
      match_pattern = input_with_three_args + set_result + name +
          ops_with_three_args + set_return;
      inplace_pattern = input_with_three_args + set_result + name + R"(_)" +
          ops_with_three_args + set_return;
    } else if (args_num == 4) {
      match_pattern = input_with_four_args + set_result + name +
          ops_with_four_args + set_return;
      inplace_pattern = input_with_four_args + set_result + name + R"(_)" +
          ops_with_four_args + set_return;
    }
    rewriter_aten_inplace.RegisterRewritePattern(
        match_pattern, inplace_pattern);
  }

  rewriter_aten_inplace.runOnGraph(graph, filter_inplace);
}

void replaceInplaceOpsWithOutplaceOps(std::shared_ptr<Graph>& graph, Block* b) {
  for (auto i = b->nodes().begin(); i != b->nodes().end();) {
    Node* n = *i;
    i++;
    for (Block* block : n->blocks()) {
      replaceInplaceOpsWithOutplaceOps(graph, block);
    }
    bool is_support = false;
    for (auto it = aten_ops_args_mapping.begin();
         it != aten_ops_args_mapping.end();
         it++) {
      if (std::string(n->kind().toQualString()).compare(it->first + "_") == 0) {
        is_support = true;
        break;
      }
    }
    if (!is_support) {
      continue;
    }
    Value* mutated_value = n->inputs().at(0);
    Value* output = n->outputs().at(0);

    // always get the latest aliasdb
    std::unique_ptr<AliasDb> aliasdb = std::make_unique<AliasDb>(graph);
    if (maybeAliveAfterNode(aliasdb.get(), n, mutated_value, output)) {
      continue;
    }

    // Do the same check as replaceOpsWithAtenInplaceOps to make sure the
    // replacement resumes after fusion fails.
    auto inputDtype = mutated_value->type()->expect<TensorType>()->scalarType();
    auto outputDtype = output->type()->expect<TensorType>()->scalarType();
    auto schema_name = n->schema().name();
    auto new_schema = schema_name.substr(0, schema_name.size() - 1);
    bool check_dtype = activation_type_promotion_mapping.find(
                           Symbol::fromQualString(new_schema)) !=
            activation_type_promotion_mapping.end()
        ? activation_type_promotion_mapping.at(
              Symbol::fromQualString(new_schema))
        : false;
    if (check_dtype &&
        (!inputDtype || !outputDtype ||
         inputDtype.value() != outputDtype.value())) {
      continue;
    }
    if (n->input(0)->uses().size() > 1) {
      continue;
    }
    if (hasSideEffectOrAlias(mutated_value, aliasdb.get())) {
      continue;
    }

    Node* new_node = graph->create(Symbol::fromQualString(new_schema), 1);
    new_node->copyMetadata(n);
    new_node->insertBefore(n);
    for (Value* input : n->inputs()) {
      new_node->addInput(input);
    }
    new_node->output()->setType(n->output()->type());
    n->output()->replaceAllUsesWith(new_node->output());
    n->destroy();
  }
}

void replaceInplaceOpsWithOutplaceOps(std::shared_ptr<Graph>& graph) {
  replaceInplaceOpsWithOutplaceOps(graph, graph->block());
}

} // namespace graph_rewrite
} // namespace jit
} // namespace torch_ipex
