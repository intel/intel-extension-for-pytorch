#include "lift_up_quant.h"
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include "utils.h"

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

bool usedBySingleOp(Value* v) {
  return v->uses().size() == 1;
}

class QuantLifter {
 private:
  std::shared_ptr<Graph> graph_;

 public:
  QuantLifter(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

  bool analyze(Block* block) {
    bool changed = false;
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      auto node = *it;

      if (node->kind() != Symbol::aten("quantize_per_tensor") &&
          node->kind() != aten::to) {
        continue;
      }

      // TODO: only supported nb_uses to be 1 for now
      auto* output_value = node->output(0);
      auto& uses = output_value->uses();
      if (uses.size() != 1) {
        continue;
      }

      auto user = uses[0].user;
      auto target = node;

      auto prev_node = node->input(0)->node();

      bool could_lift_up = true;
      while (could_lift_up) {
        auto* target_value = target->input(0);
        if (utils::isViewOp(target_value->node()) &&
            (usedBySingleOp(target_value))) {
          target = target_value->node();

          // After lifting up, need to fix the output type
          auto prev_target_type =
              target->output(0)->type()->expect<TensorType>();
          auto new_scalar_type =
              node->output(0)->type()->expect<TensorType>()->scalarType();
          auto new_target_type =
              prev_target_type->withScalarType(new_scalar_type);
          target->output(0)->setType(new_target_type);
        } else {
          could_lift_up = false;
        }
      }

      // No possible lift up, return directly
      if (target == node) {
        continue;
      }

      // From:
      // linear -> view (target) -> permute -> transpose -> to (node) -> quant
      // To:
      // linear -> to -> view (target) -> permute -> transpose -> quant
      // Finally:
      // linear -> to -> quant -> view -> permute -> transpose
      WithInsertPoint guard(target);
      auto g = target->owningGraph();

      // Construct lifted up node
      std::vector<Value*> input_values;
      input_values.push_back(target->input(0));
      for (size_t i = 1; i < node->inputs().size(); i++) {
        input_values.push_back(node->input(i));
      }
      auto new_node =
          g->create(node->kind(), input_values)->insertBefore(target);

      // Fix type of the output of lifted up node
      auto insert_point_output_type =
          target->input(0)->type()->expect<TensorType>();
      auto old_node_type = node->input(0)->type()->expect<TensorType>();
      auto new_node_type =
          insert_point_output_type->withScalarType(old_node_type->scalarType());
      new_node->output(0)->setType(new_node_type);

      target->replaceInputWith(target->input(0), new_node->output(0));
      user->replaceInputWith(node->output(0), prev_node->output(0));

      it.destroyCurrent();
      changed = true;
    }
    return changed;
  }

  void run() {
    bool changed = true;
    while (changed) {
      changed = analyze(graph_->block());
    }
  }
};

void LiftUpQuant(std::shared_ptr<Graph>& graph) {
  QuantLifter(graph).run();
  EliminateDeadCode(graph);
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
