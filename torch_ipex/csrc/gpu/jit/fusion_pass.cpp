#include <string>
#include <iostream>
#include "graph_ext.h"
#include "fusion_pass.h"
#include "accelerated_ops.h"
#include <torch/csrc/utils/hash.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/passes/pass_manager.h>


using namespace torch::jit;

// XXX: move to somewhere convenient
namespace std {
template <>
struct hash<std::pair<Symbol, Symbol>> {
  size_t operator()(std::pair<Symbol, Symbol> pair) const {
    return std::hash<uint64_t>() (
        static_cast<uint64_t>(pair.first) << 32
        | static_cast<uint64_t>(pair.second));
  }
};
}

namespace torch { namespace jit {

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
  static RuleTab dpcppRules;

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

  c10::optional<Rule> isFusable(Node *curr, Node *prev) const {
    // Is it happening in our case ???
    if (curr->owningBlock() != block_)
      return c10::nullopt;

    auto choice = dpcppRules.find({prev->kind(), curr->kind()});
    if (choice != dpcppRules.end())
      return choice;

    return c10::nullopt;
  }

  void refreshAliasDb() {
    aliasDb_ = std::make_unique<AliasDb>(graph_);
  }

  Node* fuseNodes(Node *curr, Value *path, Rule rule) {
    return fuseOpsWithNewKind(curr, path, curr->owningGraph(), rule->second);
  }

  //
  // currently we only have to fold conv2d + batch_norm
  //
  bool isFoldable(Node* node, Node* prev) {
    bool foldable = (node->kind() == aten::batch_norm
        && prev->kind() == aten::conv2d);

    //
    // Check whether all the sources are constant ???
    // Does performance improve no matter we do it pre-compiling or runtime?
    //
    auto* conv2d = reinterpret_cast<NodeExt *>(prev)->cast<Conv2dNode>();
    auto* batch_norm = reinterpret_cast<NodeExt *>(node)->cast<BatchNorm2dNode>();

    foldable = foldable
      && conv2d->hasConstantParams()
      && batch_norm->hasConstantParams();
    return foldable;
  }

  Node* foldNodes(Node *conv2d, Node *batch_norm) {
    // Change weight/bias source
    auto* fold_weight = createBatchNormFoldWeight(conv2d, batch_norm);
    fold_weight->insertBefore(conv2d);
    conv2d->replaceInput(1, fold_weight->output());

    auto* fold_bias = createBatchNormFoldBias(conv2d, batch_norm);
    fold_bias->insertBefore(conv2d);
    conv2d->replaceInput(2, fold_bias->output());

    batch_norm->replaceAllUsesWith(conv2d);
    batch_norm->destroy();
    return conv2d;
  }

  Node* createBatchNormFoldWeight(Node *conv2d, Node *batch_norm) {
    auto g = conv2d->owningGraph();
    auto newNode = g->create(dpcpp::fold_weight_sym);
    newNode->setScope(conv2d->scope());

    // We need following parameters
    newNode->addInput(conv2d->input(1));  // Conv2d weights
    newNode->addInput(batch_norm->input(1)); // Batch norm weights
    newNode->addInput(batch_norm->input(4)); // running_var (delta)
    newNode->addInput(batch_norm->input(7)); // eps

    // We get meta and type from conv2d weight value
    newNode->output()->copyMetadata(conv2d->input(1));
    newNode->output()->setType(conv2d->input(1)->type());
    newNode->output()->setDebugName(conv2d->input(1)->debugName() + ".bn_folded");

    return newNode;
  }

  Node* createBatchNormFoldBias(Node *conv2d, Node *batch_norm) {
    auto g = conv2d->owningGraph();
    auto newNode = g->create(dpcpp::fold_bias_sym);
    newNode->setScope(conv2d->scope());

    // We need following information
    newNode->addInput(conv2d->input(1)); // Conv weight
    newNode->addInput(conv2d->input(2)); // Conv bias
    newNode->addInput(batch_norm->input(1)); // batch norm weight
    newNode->addInput(batch_norm->input(2)); // batch norm bias
    newNode->addInput(batch_norm->input(3)); // running_mean (mu)
    newNode->addInput(batch_norm->input(4)); // running_var (delta)
    newNode->addInput(batch_norm->input(7)); // eps

    // We get meta and type from conv2d bias value
    newNode->output()->copyMetadata(conv2d->input(2));
    newNode->output()->setType(conv2d->input(2)->type());
    newNode->output()->setDebugName(conv2d->input(2)->debugName() + ".bn_folded");

    return newNode;
  }

  bool aliasIsSafeForSquashingValue(Node *node, Value *v) {
    bool safe = false;
    auto prev = v->node();
    if (aliasDb_->moveAfterTopologicallyValid(node, prev)) {
      if (v->uses().size() == 1
          || aliasDb_->mayAlias/* mustAlias */(v, node->output())) {
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
  bool aliasIsSafeForInplaceValue(Node *node, Value *v) {
    for (auto use : v->uses())
      if (use.user->isAfter(node))
        return false;

    return true;
  }

  const FunctionSchema &matchSchemaForFusion(c10::Symbol symbol,
      Node* prev, Node* node) {
    auto ops = getAllOperatorsFor(symbol);

    for (auto& op : ops) {
      auto& schema = op->schema();
      if (schema.arguments().size()
          == prev->inputs().size() + node->inputs().size() -1
          && schema.returns().size() == node->outputs().size())
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

  bool aliasIsSafeForFusion(Node *node, Value *v, c10::optional<Rule> r) {
    bool safe = false;
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
    // 3. If there is no alias between input and output, we can only fuse the case
    // when there is only use.
    //
    // Y-merge (conv-sum-relu?)
    // 4. We aquire alias info from resulted op schema, check whether the fusion is
    // not breaking any computational semantics.
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
    // Which y is alias to x, we check whether later is equivalent to formal. The
    // params convention when we do Y-merge: arguments from both ops comes to new
    // op in topological order. So in the exmaple conv2d's inputs comes first then
    // sum's inputs (without the input which is squashed).
    //
    safe = aliasIsSafeForSquashingValue(node, v);

    //
    // Y-merge like case
    //
    if (safe && node->inputs().size() > 1) {
      TORCH_INTERNAL_ASSERT(r);
      auto rule = *r.value();
      auto& schema = matchSchemaForFusion(rule.second, v->node(), node);
      auto o_schema = node->schema();

      auto pos = v->node()->inputs().size();

      TORCH_INTERNAL_ASSERT(schema.arguments().size()
          == pos + node->inputs().size() -1);

      for (int i = 0; i < node->inputs().size(); ++ i) {
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

  std::pair<graph_node_list::iterator, bool> processNode(Node *node) {
    auto nodeExt = reinterpret_cast<NodeExt *>(node);

    Node* pos = node;
    bool changed = false;

    // no rewrite here, check all aten Ops
    if (nodeExt->isDNNLOps()) {
      //
      // Check whether we could fuse to one certain value path
      //
      for (auto *v : node->inputs()) {
        auto prev = v->node();
        auto fuseRule = isFusable(node, prev);

        // We can fuse only one path
        if (fuseRule && aliasIsSafeForFusion(node, v, fuseRule)) {
          pos = fuseNodes(node, v, fuseRule.value());
          changed = true;
          break;
        } else if (isFoldable(node, prev)
            && aliasIsSafeForSquashingValue(node, v)) {
          pos = foldNodes(prev, node);
          changed = true;
          break;
        }
      }
    }

    return std::make_pair(++pos->iterator(), changed);
}
};

// TODO: These rules should be more scalable
OpFuser::RuleTab OpFuser::dpcppRules = {
  {{aten::conv2d, aten::relu}, dpcpp::conv2d_relu_sym},
  {{aten::conv2d, Symbol::fromQualString("aten::relu_")}, dpcpp::conv2d_relu_sym},
  {{dpcpp::conv2d_sum_sym, aten::relu}, dpcpp::conv2d_sum_relu_sym},
  {{dpcpp::conv2d_sum_sym, Symbol::fromQualString("aten::relu_")}, dpcpp::conv2d_sum_relu_sym},
  {{aten::conv2d, aten::add}, dpcpp::conv2d_sum_sym},
  {{aten::conv2d, aten::add_}, dpcpp::conv2d_sum_sym},
  {{aten::mul, aten::add_}, dpcpp::mul_add_sym},
  {{Symbol::fromQualString("quantized::conv2d"), Symbol::fromQualString("quantized::add_relu")}, dpcpp::q_conv2d_sum_relu_sym},
};

void FusionPass(std::shared_ptr<Graph> &graph) {
  // Pattern based fusion was lack of alias analysis
  // ??? It may either be too conservative or too aggressive ???
  // getSubgraphRewriter().runOnGraph(graph);
  OpFuser(graph->block(), graph).run();

  // TODO: Some post processing?? ECS/EDC/Peephole???
  // std::cout<<graph->toString(true);
  ConstantPropagation(graph);
}

void InitFusionPass() {
  RegisterPass pass_3([](std::shared_ptr<Graph>& g) {
    torch::jit::FusionPass(g);
  });
}


}} // namespace torch::jit
