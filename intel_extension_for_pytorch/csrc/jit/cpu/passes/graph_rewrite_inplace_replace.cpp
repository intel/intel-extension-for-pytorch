#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/restore_mutation.h>
#include "graph_rewrite.h"
#include "graph_rewrite_utils.h"
#include "utils.h"

namespace torch {
namespace jit {
namespace graph_rewrite {

void replaceAtenOpsWithIpexInplaceOps(std::shared_ptr<Graph>& graph) {
  std::string aten_softmax = R"(
      graph(%a, %dim:int, %half_to_float:bool):
        %r = aten::softmax(%a, %dim, %half_to_float)
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
        if (!is_contiguous(input_value)) {
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
  rewriter_aten_inplace.RegisterRewritePattern(aten_softmax, ipex_softmax_);
  rewriter_aten_inplace.runOnGraph(graph, filter_inplace_for_softmax);
}

// based on the aten inplace op list:
// {PyTorch Repo}:torch/csrc/jit/passes/restore_mutation.h#L14-L31
std::string AtenInplaceOps_with_no_args[] = {
    "aten::silu",
    "aten::sigmoid",
    "aten::tanh",
    "aten::hardsigmoid",
    "aten::hardswish",
    "aten::relu6",
    "aten::relu",
    "aten::selu"};

std::string AtenInplaceOps_with_one_args[] = {"aten::celu", "aten::leaky_relu"};

std::string AtenInplaceOps_with_two_args[] = {"aten::hardtanh"};

std::string AtenInplaceOps_with_three_args[] = {"aten::elu"};

std::string AtenInplaceOps_with_four_args[] = {"aten::rrelu"};

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

        // If type promotion is allowed, then perform dtype check
        bool check_dtype = activation_type_promotion_mapping.at(node->kind());
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
  for (int i = 0; i < 8; i++) {
    std::string match_pattern = input_no_args + set_result +
        AtenInplaceOps_with_no_args[i] + ops_no_args + set_return;
    std::string inplace_pattern = input_no_args + set_result +
        AtenInplaceOps_with_no_args[i] + R"(_)" + ops_no_args + set_return;
    rewriter_aten_inplace.RegisterRewritePattern(
        match_pattern, inplace_pattern);
  }
  for (int i = 0; i < 2; i++) {
    std::string match_pattern = input_with_one_args + set_result +
        AtenInplaceOps_with_one_args[i] + ops_with_one_args + set_return;
    std::string inplace_pattern = input_with_one_args + set_result +
        AtenInplaceOps_with_one_args[i] + R"(_)" + ops_with_one_args +
        set_return;
    rewriter_aten_inplace.RegisterRewritePattern(
        match_pattern, inplace_pattern);
  }
  for (int i = 0; i < 1; i++) {
    std::string match_pattern = input_with_two_args + set_result +
        AtenInplaceOps_with_two_args[i] + ops_with_two_args + set_return;
    std::string inplace_pattern = input_with_two_args + set_result +
        AtenInplaceOps_with_two_args[i] + R"(_)" + ops_with_two_args +
        set_return;
    rewriter_aten_inplace.RegisterRewritePattern(
        match_pattern, inplace_pattern);
  }
  for (int i = 0; i < 1; i++) {
    std::string match_pattern = input_with_three_args + set_result +
        AtenInplaceOps_with_three_args[i] + ops_with_three_args + set_return;
    std::string inplace_pattern = input_with_three_args + set_result +
        AtenInplaceOps_with_three_args[i] + R"(_)" + ops_with_three_args +
        set_return;
    rewriter_aten_inplace.RegisterRewritePattern(
        match_pattern, inplace_pattern);
  }
  for (int i = 0; i < 1; i++) {
    std::string match_pattern = input_with_four_args + set_result +
        AtenInplaceOps_with_four_args[i] + ops_with_four_args + set_return;
    std::string inplace_pattern = input_with_four_args + set_result +
        AtenInplaceOps_with_four_args[i] + R"(_)" + ops_with_four_args +
        set_return;
    rewriter_aten_inplace.RegisterRewritePattern(
        match_pattern, inplace_pattern);
  }
  rewriter_aten_inplace.runOnGraph(graph, filter_inplace);
}

} // namespace graph_rewrite
} // namespace jit
} // namespace torch
