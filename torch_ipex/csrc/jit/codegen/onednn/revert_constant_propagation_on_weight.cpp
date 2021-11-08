#include <ATen/native/quantized/cpu/quant_utils.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include "jit/codegen/onednn/revert_constant_propagation_on_weight.h"

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

c10::optional<Node*> mayGetInputNode(Node* node) {
  if (node->inputs().size() > 0) {
    auto input_node = node->input(0)->node();
    return input_node;
  }
  return c10::nullopt;
}

bool inputQuantized(Node* node) {
  auto input_node = mayGetInputNode(node);
  if (input_node.has_value()) {
    if (input_node.value()->kind() == Symbol::aten("dequantize")) {
      return true;
    }
    if (input_node.value()->kind() == aten::to) {
      auto pre_input_node = mayGetInputNode(input_node.value());
      if (pre_input_node.has_value() &&
          pre_input_node.value()->kind() == Symbol::aten("dequantize")) {
        return true;
      }
    }
  }
  return false;
}

bool outputQuantized(Node* node) {
  auto output_value = node->output(0);
  auto& uses = output_value->uses();

  for (const auto& use : uses) {
    if (use.user->kind() == Symbol::aten("quantize_per_tensor")) {
      return true;
    }
  }
  return false;
}

// Need to revert ConstantPropagation on weight of conv, linear and
// embedding_bag
bool needRevertConstantPropagationOnWeight(Node* node) {
  if (node->kind() == aten::linear || node->kind() == aten::_convolution ||
      node->kind() == aten::conv2d) {
    if (inputQuantized(node)) {
      return true;
    }
  }

  if (node->kind().toQualString() == std::string("torch_ipex::embedding_bag")) {
    if (outputQuantized(node)) {
      return true;
    }
  }

  return false;
}

float getPerTensorScale(at::Tensor weight) {
  const int precision = 8;
  float min = weight.min().item<float>();
  float max = weight.max().item<float>();

  float w_max_value = std::max(std::abs(min), max);

  auto qparams = quant_utils::ChooseQuantizationParams(
      /*min*/ -w_max_value,
      /*max*/ w_max_value,
      /*q_min*/ -(1 << (precision - 1)),
      /*q_max*/ ((1 << (precision - 1)) - 1),
      /*preserve_sparsity=*/true);

  return qparams.scale;
}

at::Tensor getPerChannelScale(at::Tensor weight) {
  const int precision = 8;

  std::vector<std::vector<float>> min_max_values;
  for (int l = 0; l < weight.size(0); l++) {
    min_max_values.push_back(
        {weight[l].min().item<float>(), weight[l].max().item<float>()});
  }

  std::vector<float> w_scales;
  for (auto n = 0; n < min_max_values.size(); n++) {
    auto w_max_value =
        std::max(std::abs(min_max_values[n][0]), min_max_values[n][1]);
    auto qparams = quant_utils::ChooseQuantizationParams(
        /*min*/ -w_max_value,
        /*max*/ w_max_value,
        /*q_min*/ -(1 << (precision - 1)),
        /*q_max*/ ((1 << (precision - 1)) - 1),
        /*preserve_sparsity=*/true);
    w_scales.push_back(qparams.scale);
  }

  return at::tensor(w_scales, at::device(at::kCPU).dtype(at::kDouble));
}

Node* insertConstantQuantizedWeightWithDequant(
    Node* insertPointNode,
    TypePtr output_type,
    at::Tensor q_weight) {
  WithInsertPoint guard(insertPointNode);
  auto g = insertPointNode->owningGraph();

  Node* q_weight_node = g->create(prim::Constant);

  q_weight_node->output()->inferTypeFrom(q_weight);
  q_weight_node->t_(attr::value, std::move(q_weight));

  g->insertNode(q_weight_node);

  auto dequantize_node =
      g->create(Symbol::aten("dequantize"), {q_weight_node->output()})
          ->insertAfter(q_weight_node);
  dequantize_node->output()->setType(output_type);
  return dequantize_node;
}

Node* insertTypeCast(
    Node* insertPointNode,
    TypePtr output_type,
    Node* prev_node) {
  WithInsertPoint guard(insertPointNode);
  auto g = insertPointNode->owningGraph();

  // hard-code here to construct a TypeCast node:
  // aten::to(ScalarType=at::ScalarType::BFloat16, non_blocking=False,
  // copy=False, memory_format=None)
  auto* scalar_type_constant =
      g->prependNode(g->create(prim::Constant))
          ->i_(
              Symbol::attr("value"),
              static_cast<int64_t>(at::ScalarType::BFloat16));
  scalar_type_constant->output()->setType(IntType::get());

  auto* non_blocking_constant = g->prependNode(g->create(prim::Constant))
                                    ->i_(Symbol::attr("value"), false);
  non_blocking_constant->output()->setType(BoolType::get());

  auto* copy_constant = g->prependNode(g->create(prim::Constant))
                            ->i_(Symbol::attr("value"), false);
  copy_constant->output()->setType(BoolType::get());

  auto* memory_format_constant = g->prependNode(g->create(prim::Constant));
  memory_format_constant->output()->setType(NoneType::get());

  auto to_bf16_node = g->insertNode(g->create(
      aten::to,
      {prev_node->output(),
       scalar_type_constant->output(),
       non_blocking_constant->output(),
       copy_constant->output(),
       memory_format_constant->output()}));
  to_bf16_node->output()->setType(output_type);

  return to_bf16_node;
}

void WeightConstantPropagationReversion(Node* n) {
  if (needRevertConstantPropagationOnWeight(n)) {
    bool is_embedding_bag =
        n->kind().toQualString() == std::string("torch_ipex::embedding_bag");

    // For node torch_ipex::embedding_bag, weight is at input(0)
    // For node conv and linear, weight is at input(1)
    auto weight_index = is_embedding_bag ? 0 : 1;
    auto weight_node = n->input(weight_index)->node();
    if (weight_node->kind() != prim::Constant) {
      return;
    }

    Value* v = weight_node->output();
    if (!(v->type()->cast<TensorType>())) {
      return;
    }

    auto weight_value_type = v->type()->expect<TensorType>();

    auto weight = toIValue(v)->toTensor();
    auto scalar_type = weight.scalar_type();

    if (is_embedding_bag) {
      if (scalar_type == at::ScalarType::Float) {
        // For embedding_bag: hard-coded to be per_tensor and symmetric
        float w_scales = getPerTensorScale(weight);
        auto q_weight = at::quantize_per_tensor(
            weight, w_scales, /*zero_point*/ 0, at::kQInt8);

        auto dequantize_node = insertConstantQuantizedWeightWithDequant(
            n,
            weight_value_type->withScalarType(at::ScalarType::Float),
            q_weight);

        n->replaceInputWith(n->input(0), dequantize_node->output());
      }
    } else {
      if (scalar_type == at::ScalarType::Float ||
          scalar_type == at::ScalarType::BFloat16) {
        bool bf16 = scalar_type == at::kBFloat16;
        weight = bf16 ? weight.to(at::ScalarType::Float) : weight;

        // For conv and linear: hard-coded to be per_channel and symmetric
        at::Tensor weight_scale = getPerChannelScale(weight);
        auto zero_point = at::zeros(weight_scale.numel(), at::dtype(at::kLong));

        auto q_weight = at::quantize_per_channel(
            weight,
            weight_scale,
            zero_point,
            /*axis*/ 0,
            at::kQInt8);

        auto dequantize_node = insertConstantQuantizedWeightWithDequant(
            n,
            weight_value_type->withScalarType(at::ScalarType::Float),
            q_weight);

        if (!bf16) {
          n->replaceInputWith(n->input(1), dequantize_node->output());
        } else {
          auto to_bf16_node = insertTypeCast(
              n,
              weight_value_type->withScalarType(at::ScalarType::BFloat16),
              dequantize_node);

          n->replaceInputWith(n->input(1), to_bf16_node->output());
        }
      }
    }
  }
}

void WeightConstantPropagationReversion(at::ArrayRef<Block*> blocks) {
  for (Block* block : blocks)
    for (Node* node : block->nodes())
      WeightConstantPropagationReversion(node);
}

void RevertConstantPropagationOnWeight(std::shared_ptr<Graph>& graph) {
  WeightConstantPropagationReversion(graph->block());
  EliminateDeadCode(graph);
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch