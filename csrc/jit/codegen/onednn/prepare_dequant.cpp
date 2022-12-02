#include "prepare_dequant.h"
#include "operator.h"
#include "utils.h"

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

using namespace torch::jit;

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
          Symbol::aten("quantize_per_channel"),
          prim::Constant}},
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

void addInformationForDequant(Node* node, Node* input_node) {
  if (input_node->kind() == Symbol::aten("quantize_per_tensor")) {
    node->s_(Symbol::attr("qtype"), std::string("per_tensor"));

    std::vector<int64_t> zps_vector = Operator::IntToVector(input_node, 2);
    node->is_(Symbol::attr("zps"), zps_vector);

    double scale = Operator::Float(input_node, 1);
    node->fs_(Symbol::attr("scales"), {scale});
  } else if (input_node->kind() == Symbol::aten("quantize_per_channel")) {
    node->s_(Symbol::attr("qtype"), std::string("per_channel"));
    node->t_(Symbol::attr("zps"), Operator::Tensor(input_node, 2));
    node->t_(Symbol::attr("scales"), Operator::Tensor(input_node, 1));
    node->i_(Symbol::attr("axis"), Operator::Int(input_node, 3));
  } else {
    TORCH_CHECK(
        input_node->kind() == prim::Constant,
        "Expect input_node kind to be prim::Constant but got ",
        input_node->kind().toQualString());

    Value* v = input_node->output();
    TORCH_CHECK(
        v->type()->cast<TensorType>(),
        "Constant input to dequant must be Tensor type");
    auto qtensor = toIValue(v)->toTensor();

    TORCH_CHECK(
        qtensor.scalar_type() == at::ScalarType::QInt8 ||
            qtensor.scalar_type() == at::ScalarType::QUInt8,
        "Expect input to dequant to be int8 dtype but got ",
        qtensor.scalar_type());
    auto scalar_type = qtensor.scalar_type();

    switch (qtensor.qscheme()) {
      case at::kPerTensorAffine:
        node->s_(Symbol::attr("qtype"), std::string("per_tensor"));
        node->is_(
            Symbol::attr("zps"),
            Operator::IntValueToVector(qtensor.q_zero_point()));
        node->fs_(Symbol::attr("scales"), {qtensor.q_scale()});
        break;
      case at::kPerChannelAffine:
        node->s_(Symbol::attr("qtype"), std::string("per_channel"));
        node->t_(Symbol::attr("zps"), qtensor.q_per_channel_zero_points());
        node->t_(Symbol::attr("scales"), qtensor.q_per_channel_scales());
        node->i_(Symbol::attr("axis"), qtensor.q_per_channel_axis());
        break;
      default:
        TORCH_CHECK(
            false,
            "Unsupported tensor quantization type ",
            toString(qtensor.qscheme()));
    }
  }
}

void DequantInformationSave(Node* node) {
  if (node->kind() != Symbol::aten("dequantize")) {
    return;
  }
  if (node->numAttributes() != 0) {
    return;
  }
  Node* input_node = node->input(0)->node();
  if (!utils::isSupportedAsInputToDequant(input_node)) {
    return;
  }
  addInformationForDequant(node, input_node);
}

void DequantInformationSave(at::ArrayRef<Block*> blocks) {
  for (Block* block : blocks)
    for (Node* node : block->nodes())
      DequantInformationSave(node);
}

void SaveDequantInformation(std::shared_ptr<Graph>& graph) {
  DequantInformationSave(graph->block());
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch_ipex
