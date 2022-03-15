#include "fusion_pass.h"
#include <string>
#include "autocast/autocast_mode.h"
#include "codegen/onednn/interface.h"
#include "cpu/passes/graph_rewrite.h"
#include "cpu/passes/prepack_folding.h"

#include "cpu/kernels/Matmul.h"
#include "cpu/passes/concat_linear.h"
#include "cpu/passes/frozen_conv_folding.h"
#include "cpu/passes/frozen_linear_folding.h"

#include <c10/util/hash.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/batch_mm.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/operator.h>

using namespace torch::jit;

// XXX: move to somewhere convenient
namespace std {
template <>
struct hash<std::pair<Symbol, Symbol>> {
  size_t operator()(std::pair<Symbol, Symbol> pair) const {
    return std::hash<uint64_t>()(
        static_cast<uint64_t>(pair.first) << 32 |
        static_cast<uint64_t>(pair.second));
  }
};
} // namespace std

namespace torch {
namespace jit {
//
// The main goal of MKL-DNN fusion is to limit bandwidth wasting.
// MKL-DNN provided post ops to fuse ops in its output stage
// What we could do is listed inside RuleTab.
//
class OpFuser {
  Block* block_;
  std::unique_ptr<AliasDb> aliasDb_;
  std::shared_ptr<Graph> graph_;
  using Symbols = std::vector<Symbol>;
  using RuleTab = std::unordered_map<::std::pair<Symbol, Symbol>, Symbol>;
  using Rule = RuleTab::iterator;
  static RuleTab dnnlRules;

 public:
  OpFuser(Block* block, std::shared_ptr<Graph> graph)
      : block_(block), graph_(std::move(graph)) {}

  void run() {
    bool any_changed = true;
    while (any_changed) {
      any_changed = false;
      refreshAliasDb();
      for (auto it = block_->nodes().begin(); it != block_->nodes().end();) {
        bool changed;
        std::tie(it, changed) = processNode(*it);
        any_changed |= changed;
      }
    }

    refreshAliasDb();

    for (Node* node : block_->nodes()) {
      for (Block* sub : node->blocks()) {
        OpFuser(sub, graph_).run();
      }
    }
  }

  c10::optional<Rule> isFusable(Node* curr, Node* prev) const {
    // Is it happening in our case ???
    if (curr->owningBlock() != block_)
      return c10::nullopt;

    auto choice = dnnlRules.find({prev->kind(), curr->kind()});
    if (choice != dnnlRules.end())
      return choice;

    return c10::nullopt;
  }

  void refreshAliasDb() {
    aliasDb_ = std::make_unique<AliasDb>(graph_);
  }

  Node* fuseOpsWithNewKind(Node* curr, Value* v, Graph* g, NodeKind kind) {
    auto newNode = g->create(kind);
    auto prev = v->node();
    newNode->insertBefore(prev);
    newNode->setScope(prev->scope());
    newNode->copyAttributes(*prev);

    for (auto input : prev->inputs()) {
      newNode->addInput(input);
    }

    for (auto input : curr->inputs()) {
      if (input != v) {
        newNode->addInput(input);
      }
    }

    // Copy curr or prev?
    newNode->output()->copyMetadata(prev->output());
    newNode->output()->setType(prev->output()->type());

    v->replaceAllUsesWith(newNode->output());
    curr->replaceAllUsesWith(newNode);

    prev->destroy();
    curr->destroy();

    return newNode;
  }

  Node* fuseNodes(Node* curr, Value* path, Rule rule) {
    return fuseOpsWithNewKind(curr, path, curr->owningGraph(), rule->second);
  }

  bool aliasIsSafeForSquashingValue(Node* node, Value* v) {
    bool safe = false;
    auto prev = v->node();
    if (aliasDb_->moveAfterTopologicallyValid(node, prev)) {
      if (v->uses().size() == 1 ||
          aliasDb_->mayAlias /* mustAlias */ (v, node->output())) {
        safe = true;
      }
    }
    return safe;
  }

  //
  // Check whether we could change specific input to be inplace with output
  // Any use topologically after node will fail it.
  // XXX: haven't considered loop
  //
  bool aliasIsSafeForInplaceValue(Node* node, Value* v) {
    for (auto use : v->uses())
      if (use.user->isAfter(node))
        return false;

    return true;
  }

  const FunctionSchema& matchSchemaForFusion(
      c10::Symbol symbol,
      Node* prev,
      Node* node) {
    auto ops = getAllOperatorsFor(symbol);

    for (auto& op : ops) {
      auto& schema = op->schema();
      if (schema.arguments().size() ==
              prev->inputs().size() + node->inputs().size() - 1 &&
          schema.returns().size() == node->outputs().size())
        return schema;
    }

    // throw
    auto er = ErrorReport(node->sourceRange());
    er << "Schema not found for fusion process. \n";
    er << "Prev: " << *prev << "\n";
    er << "Node: " << *node << "\n";

    if (ops.size() > 0) {
      er << "\ncandidates were:\n";
      for (auto& op : ops)
        er << "  " << op->schema() << "\n";
    } else {
      er << "\nno candidates found\n";
    }
    er << "within the graph:\n";
    er << *node->owningGraph() << "\n";
    throw er;
  }

  bool aliasIsSafeForFusion(Node* node, Value* v, c10::optional<Rule> r) {
    bool safe = false;
    // Returns false if the two nodes to be fused do not have the same owning
    // block
    if (node->owningBlock() != v->node()->owningBlock()) {
      return safe;
    }
    // TODO: it might be flawed because we don't have 'alias must' information
    //
    // Simple fusion, unary ops:
    // Example: conv2d -> relu to conv2d_relu
    //
    // To maintain equivalence before and after fusion, we have some rules:
    // 1. Op could be moved safely right after the op it fuse to.
    // 2. If one of node's input and output are alias must (relu_?), we could
    // replace all uses of input to use output, which remove the use that might
    // clogging the fuse path which is to be squashed.
    // 3. If there is no alias between input and output, we can only fuse the
    // case when there is only use.
    //
    // Y-merge (conv-sum-relu?)
    // 4. We aquire alias info from resulted op schema, check whether the fusion
    // is not breaking any computational semantics.
    //
    // A Y-merge fusion, like:
    //           conv2d_inputs | or | conv2d_inputs
    //             /           |    |      \
    //      x   conv2d         |    |    conv2d  x
    //       \   /             |    |        \  /
    //        add              |    |        add
    //         |               |    |         |
    //         y               |    |         y
    //
    // both to:
    //
    // conv2d_inputs  x(a!)
    //      \        /
    //     conv2d_sum
    //         |
    //       y(a!)
    //
    // Which y is alias to x, we check whether later is equivalent to formal.
    // The params convention when we do Y-merge: arguments from both ops comes
    // to new op in topological order. So in the exmaple conv2d's inputs comes
    // first then sum's inputs (without the input which is squashed).
    //
    safe = aliasIsSafeForSquashingValue(node, v);

    //
    // Y-merge like case
    //
    if (safe && node->inputs().size() > 1) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(r);
      auto rule = *r.value();
      auto& schema = matchSchemaForFusion(rule.second, v->node(), node);
      auto o_schema = node->schema();

      auto pos = v->node()->inputs().size();

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          schema.arguments().size() == pos + node->inputs().size() - 1);

      for (int i = 0; i < node->inputs().size(); ++i) {
        if (node->input(i) != v) { /* avoid squashing path */
          auto aliasInfo = schema.arguments()[pos++].alias_info();
          if (!aliasInfo)
            continue;

          // Introdued new alias write to
          if (aliasInfo->isWrite()) {
            auto old_info = o_schema.arguments()[i].alias_info();
            if (!old_info || !old_info->isWrite()) {
              // Introduced new written to alias
              safe = safe && aliasIsSafeForInplaceValue(node, node->input(i));
            }
          }
        }
      }

      // XXX: Do we have to handle output alias change case?
    }
    return safe;
  }

  std::pair<graph_node_list::iterator, bool> processNode(Node* node) {
    Node* pos = node;
    bool changed = false;

    //
    // Check whether we could fuse to one certain value path
    //
    for (auto* v : node->inputs()) {
      auto prev = v->node();
      auto fuseRule = isFusable(node, prev);

      // We can fuse only one path
      if (fuseRule && aliasIsSafeForFusion(node, v, fuseRule)) {
        pos = fuseNodes(node, v, fuseRule.value());
        changed = true;
        break;
      }
    }
    return std::make_pair(++pos->iterator(), changed);
  }
};

// TODO: These rules should be more scalable
OpFuser::RuleTab OpFuser::dnnlRules = {
    {{aten::matmul, aten::div}, ipex::matmul_div},
};

// Including in-place optimizations that try to (conditionally)
// replace the origin op with in-place opted one for better performance.
// This in-place optimized ops may come from either oneDNN or aten
void ApplyInplaceOptimization(std::shared_ptr<Graph>& graph) {
  // try to replace aten ops with ipex in-place ops
  graph_rewrite::replaceAtenOpsWithIpexInplaceOps(graph);
  // try to replace aten ops with aten in-place ops
  graph_rewrite::replaceOpsWithAtenInplaceOps(graph);
}

void RemoveBailOutNodesAndSpecializeTypes(Block* b) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    if (it->kind() == prim::BailOut) {
      it->output()->replaceAllUsesWith(it->inputs()[1]);
      auto profiled_type = it->output()->type()->expect<TensorType>();

      if (profiled_type == TensorType::get()) {
        continue;
      }

      auto input_type = it->inputs()[1]->type()->expect<TensorType>();
      if (input_type == TensorType::get()) {
        it->inputs()[1]->setType(profiled_type);
      } else {
        it->inputs()[1]->setType(input_type->merge(*profiled_type));
      }
      it.destroyCurrent();

    } else {
      for (Block* ib : it->blocks()) {
        RemoveBailOutNodesAndSpecializeTypes(ib);
      }
    }
  }
}

void RemoveBailoutTemplateNodes(Block* b) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    if (it->kind() == prim::BailoutTemplate) {
      it.destroyCurrent();
    } else {
      for (Block* ib : it->blocks()) {
        RemoveBailoutTemplateNodes(ib);
      }
    }
  }
}

void IPEXFusionPass(std::shared_ptr<Graph>& graph) {
  // remove dropout;
  torch::jit::removeDropout(graph);

  // concat multi-linear with same input
  FrozenConcatLinear(graph);

  // Fuse the scores calculation(dim + matmul + (add)? + softmax) for
  // Multi-Head-Attention
  graph_rewrite::FuseMHAScoreCalc(graph);

  // Replace _convolution with conv2d or conv3d
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  // convolution folding
  FrozenConvFolding(graph);

  // convolution fusion
  graph_rewrite::insertPrePackedConvOp(graph);
  graph_rewrite::fuseConvWithEltwise(graph);
  graph_rewrite::fuseConvAddRelu(graph);
  graph_rewrite::fuseBottleneck(graph);

  // linear folding
  FoldFrozenLinearAddOrSub(graph);
  FoldFrozenLinearMulOrDiv(graph);

  // linear fusion
  graph_rewrite::insertPrePackedLinearOp(graph);
  graph_rewrite::fuseLinearWithEltwise(graph);
  graph_rewrite::fuseLinearAddRelu(graph);

  graph_rewrite::FuseLinearSwishCustomized(graph);
  // fuse add+layernorm
  graph_rewrite::FuseAddLayerNorm(graph);
  // deconvolution fusion
  graph_rewrite::insertPrePackedConvTranspose2dOp(graph);

  // fuse concat+bn+relu for the input float tensors with the same sizes
  // and channelslast format
  // hence the concat dim should be the channel
  graph_rewrite::FuseConcatBnRelu(graph);

  // Fuse operators as shuffle
  graph_rewrite::FuseShuffle(graph);
  // Pattern based fusion was lack of alias analysis
  // ??? It may either be too conservative or too aggressive ???
  // getSubgraphRewriter().runOnGraph(graph);
  OpFuser(graph->block(), graph).run();

  // apply inplace optimization before outplace op replacements
  ApplyInplaceOptimization(graph);

  // replace aten max_pool2d with ipex max_pool2d
  graph_rewrite::replaceAtenMaxPool2dWithIpexMaxPool2d(graph);

  // replace aten softmax with ipex softmax
  graph_rewrite::replaceAtenSoftmaxWithIpexSoftmax(graph);

  // replace aten::batch_norm with ipex::batch_norm, it will be removed
  // after TensorExprs fix the performance issue(IPB-808).
  graph_rewrite::replaceAtenBatchNormWithIpexBatchNorm(graph);
  // TODO: Some post processing?? ECS/EDC/Peephole???
  ConstantPropagation(graph);
  // folding prepacking ops.
  PrePackingOpsFolder(graph);
}

bool checkQuantization(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      checkQuantization(sub);
    }

    if (node->kind() == Symbol::aten("quantize_per_tensor") ||
        node->kind() == Symbol::aten("dequantize") ||
        node->kind() == Symbol::aten("quantize_per_channel")) {
      return true;
    }
  }
  return false;
}

bool isQuantized(const std::shared_ptr<Graph>& graph) {
  return checkQuantization(graph->block());
}

void FusionPass(std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP(
      "Before RemoveProfileNodesAndSpecializeTypes. Beginning of "
      "optimization pass",
      graph);
  RemoveProfileNodesAndSpecializeTypes(graph);

  // remove BailOut and BailoutTemplate
  RemoveBailOutNodesAndSpecializeTypes(graph->block());
  RemoveBailoutTemplateNodes(graph->block());

  // LLGA fusion pass for int8
  GRAPH_DUMP(
      "After RemoveProfileNodesAndSpecializeTypes. Before LLGA fusion pass",
      graph);
  if (isQuantized(graph) || torch_ipex::autocast::is_llga_fp32_bf16_enabled()) {
    fuser::onednn::fuseGraph(graph);
  }
  GRAPH_DUMP("After LLGA fusion pass. Before IPEXFusionPass", graph);

  // IPEX fusion pass for fp32 and bf16
  IPEXFusionPass(graph);
  GRAPH_DUMP(
      "After IPEXFusionPass. Before RemoveTensorTypeSpecializations", graph);

  // TODO: workaround here to go throughput the TE fuser pass before
  // RemoveTensorTypeSpecializations since TE fuser needs the type
  // specializations
  LowerSimpleTuples(graph);
  BatchMM(graph);

  if (tensorExprFuserEnabled()) {
    FuseTensorExprs(graph, getFusionGroupInlining() ? 2 : 1);
  }

  RemoveTensorTypeSpecializations(graph);
  GRAPH_DUMP(
      "After RemoveTensorTypeSpecializations. End of optimization pass", graph);
}
} // namespace jit
} // namespace torch
