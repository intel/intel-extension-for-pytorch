#include "graph_ext.h"
#include "accelerated_ops.h"
#include "op_rewrite.h"
#include "format_analysis.h"

#include <torch/csrc/utils/memory.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>


namespace torch { namespace jit {

//
// Optimization, since dnnl ops could understand detailed formats after
// computational intensive operator:
//
//      f1: conv (f0)
//      f1.reorder: reorder(f1)
//      f2: dnnl_ops (f1.reorder)
//      =>
//      f1: conv (f0)
//      f1.reorder: reorder(f1) <--- DCE would delete it if no one use it
//      f2: dnnl_ops (f1)
//      f2.reorder: reorder(f2)
//
// Do this until we move all the MKL-DNN ops inside reorder bracket
//
class FormatAnalyzer {
private:
  std::unique_ptr<AliasDb> aliasDb_;
  Block *block_;
  std::shared_ptr<Graph> graph_;
public:
  FormatAnalyzer(Block *block, std::shared_ptr<Graph> graph)
    : block_(block), graph_(std::move(graph)) {}

  void refreshAliasDb () {
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
  }

  //
  // Interface to garantee no one may write dnnl_op's input set
  //
  bool hasWritersToInputsTopologicallyBetween(graph_node_list_iterator begin,
      graph_node_list_iterator end, const ValueSet set) {
    bool had = false;

    if (begin->isBefore(*end)) {
      for(auto it = begin; it != end; ++ it) {
        had |= aliasDb_->writesToAlias(*it, set);
      }
    }

    // ??? Shall we consider wildcard

    return had;
  }

  //
  // For supported ops like relu/batch_norm etc.
  // adjust op receive output before preceding reorder
  //
  bool bypassReorder(Node* op, Value* input) {
    // Assert here
    auto adjusted = false;
    auto reorder = input->node();

    if (!hasWritersToInputsTopologicallyBetween(++reorder->iterator(),
          op->iterator(), {input})) {
      auto opExt = reinterpret_cast<NodeExt *>(op);
      auto reorderExt = reinterpret_cast<NodeExt *>(reorder);

      opExt->replaceInputWith(input, reorderExt->input());
      opExt->appendReorder(natureFormat);

      adjusted = true;
    }

    return adjusted;
  }

  //
  // merge reorder, kind like move op before reorder except we don't append new
  // reorder
  //
  bool mergeReorder(Node *op, Node *reorder) {
    auto merged = false;

    if (!hasWritersToInputsTopologicallyBetween(++reorder->iterator(),
          op->iterator(), {op->input()})) {
      auto reorder_second = reinterpret_cast<NodeExt *>(op);
      auto reorder_first = reinterpret_cast<NodeExt *>(reorder);

      if (reorder_first->outputFormat() == natureFormat
          && reorder_second->inputFormat() == formatTag::any
          && reorder_second->outputFormat() == natureFormat) {
        // bypass first reorder
        reorder_second->replaceInput(0, reorder_first->input());
        reorder_first->output()->replaceAllUsesWith(reorder_second->output());
      } else {
        //
        // Otherwise use second reorder
        // TODO: review this path
        //
        reorder_second->replaceInput(0, reorder_first->input());
      }

      merged = true;
    }

    return merged;
  }

  bool analyzeNode(Node *node) {
    auto changed = false;
    auto n = reinterpret_cast<NodeExt *>(node);

    if (n ->isDNNLOps()) {
      for (auto *input : n->inputs()) {
        auto upstream = reinterpret_cast<NodeExt *>(input->node());
        if (upstream->isReorder()) {
          if (n ->isReorder())
            changed = mergeReorder(n, upstream);
          else
            changed = bypassReorder(n, input);
        }
      }
    }

    return changed;
  }

  void run() {
    bool changed = true;
    while (changed) {
      changed = false;
      refreshAliasDb();

      for(Node* node : block_->nodes()) {
        changed |= analyzeNode(node);
      }
    }

    for(Node* node : block_->nodes()) {
      for (Block* subBlock : node->blocks()) {
        FormatAnalyzer(subBlock, graph_).run();
      }
    }
  }
};

void FormatOptimize(std::shared_ptr<Graph>& graph) {
  FormatAnalyzer(graph->block(), graph).run();

  // XXX: Do we need to do dead code elimination here?
  EliminateDeadCode(graph->block());
}
}} // namespace torch::jit
