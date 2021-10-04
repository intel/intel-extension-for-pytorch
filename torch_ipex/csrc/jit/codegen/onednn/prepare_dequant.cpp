#include "jit/codegen/onednn/prepare_dequant.h"

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

class OpSplitter {
 private:
  std::shared_ptr<Graph> graph_;

 public:
  OpSplitter(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

  bool analyzeNode(Node* node) {
    // If node->kind() matches the NodeKind, the node will be a candidate to be
    // splitted. If the input to the current node matches with InputKind, will
    // split the node
    static std::unordered_map<Symbol, std::set<Symbol>> NodeKindToInputKind{
        {aten::to, {Symbol::aten("dequantize")}},
        {Symbol::aten("dequantize"),
         {Symbol::aten("quantize_per_tensor"),
          Symbol::aten("quantize_per_channel")}},
    };

    auto it = NodeKindToInputKind.find(node->kind());
    if (it == NodeKindToInputKind.end()) {
      return false;
    }

    auto& input_kind = it->second;
    if (input_kind.find(node->input(0)->node()->kind()) == input_kind.end()) {
      return false;
    }

    auto* output_value = node->output(0);
    auto& uses = output_value->uses();
    int nb_uses = uses.size();
    if (nb_uses < 2) {
      return false;
    }

    WithInsertPoint guard(node);
    auto g = node->owningGraph();

    // save the users before modifying the graph
    std::vector<torch::jit::Node*> output_users;
    for (const auto& use : uses) {
      output_users.push_back(use.user);
    }

    auto output_type = output_value->type()->expect<TensorType>();

    std::vector<NamedValue> input_values;
    for (auto* v : node->inputs()) {
      NamedValue nv(v);
      input_values.push_back(nv);
    }
    at::ArrayRef<NamedValue> args(input_values);

    for (int i = 1; i < nb_uses; i++) {
      auto split_node = g->insert(node->kind(), args);
      split_node->setType(output_type);

      auto output_user = output_users[i];
      output_user->replaceInputWith(output_value, split_node);
    }
    return true;
  }

  void run() {
    bool changed = true;
    while (changed) {
      changed = false;
      for (Node* node : graph_->block()->nodes()) {
        changed |= analyzeNode(node);
      }
    }
  }
};

void PrepareDequantForLLGA(std::shared_ptr<Graph>& graph) {
  OpSplitter(graph).run();
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
