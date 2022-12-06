#include "fusion_pass.h"
#include <c10/util/hash.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <iostream>
#include <string>
#include <vector>
#include "accelerated_ops.h"
using namespace torch::jit;
using namespace std;

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

namespace {
/*IPEX_DEFINE_CONV_FUSION

This macro will convinently generate the rule of conv related fusion pattern.
This macro can be adopt when the fusion symbol have aten::post-op form in jit
ir, and the symbol of fusion op is defined with variabe named: conv2d_op_sym,
_convolution_op_sym, q_conv2d_op_sym, this is aligned with the defination of
macro IPEX_GENERAL_CONV_SYMBOL_DECLARATION. The expansion of this macro should
be:

{{q_conv2d_sym, Symbol::fromQualString("aten::op")},
   xpu::q_conv2d_op_sym},
{{aten::conv2d, Symbol::fromQualString("aten::op")},
    xpu::conv2d_op_sym},
{{_conv_sym, Symbol::fromQualString("aten::op")},
    xpu::_convolution_op_sym}

and for INT8 scenario, aten::dequantize might be inserted between q_conv2d and
post-op. we can adopt IPEX_DEFINE_CONV_FUSION_DEQUANTIZE instead to handle this
kind of cases which will automatically fuse the aten::dequantize.
*/
// TODO: verify the accuracy of quantized convolution with post op fusion and
// enable it in jit
#define IPEX_DEFINE_CONV_FUSION(func)                              \
  {{aten::conv2d, Symbol::fromQualString("aten::" #func)},         \
   xpu::conv2d_##func##_sym},                                      \
      {{_conv_sym, Symbol::fromQualString("aten::" #func)},        \
       xpu::_convolution_##func##_sym},                            \
      {{aten::conv2d, Symbol::fromQualString("aten::" #func "_")}, \
       xpu::conv2d_##func##_sym},                                  \
  {                                                                \
    {_conv_sym, Symbol::fromQualString("aten::" #func "_")},       \
        xpu::_convolution_##func##_sym                             \
  }
// {                                                         \
  //   {q_conv2d_sym, Symbol::fromQualString("aten::" #func)}, \
  //       xpu::q_conv2d_##func##_sym                          \
  // }

#define IPEX_DEFINE_CONV_FUSION_WITH_DEQUANTIZE(func)              \
  {{aten::conv2d, Symbol::fromQualString("aten::" #func)},         \
   xpu::conv2d_##func##_sym},                                      \
      {{_conv_sym, Symbol::fromQualString("aten::" #func)},        \
       xpu::_convolution_##func##_sym},                            \
      {{aten::conv2d, Symbol::fromQualString("aten::" #func "_")}, \
       xpu::conv2d_##func##_sym},                                  \
  {                                                                \
    {_conv_sym, Symbol::fromQualString("aten::" #func "_")},       \
        xpu::_convolution_##func##_sym                             \
  }
// {                                                                    \
  //   {q_conv2d_dequantize_sym, Symbol::fromQualString("aten::" #func)}, \
  //       xpu::q_conv2d_##func##_sym                                     \
  // }

#define IPEX_DEFINE_LINEAR_FUSION(func)                         \
  {{aten::linear, Symbol::fromQualString("aten::" #func)},      \
   xpu::linear_##func##_sym},                                   \
  {                                                             \
    {aten::linear, Symbol::fromQualString("aten::" #func "_")}, \
        xpu::linear_##func##_sym                                \
  }

} // namespace

namespace torch {
namespace jit {
namespace xpu {

vector<c10::Symbol> not_check_uses_ops{xpu::mish_compound_sym};

//
// The main goal of oneDNN fusion is to limit bandwidth wasting.
// oneDNN provided post ops to fuse ops in its output stage
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

  bool BatchNorm2dNode_hasConstantParams(Node* node) const {
    bool has = node->input(1)->node()->kind() == prim::Constant &&
        node->input(2)->node()->kind() == prim::Constant &&
        node->input(3)->node()->kind() == prim::Constant &&
        node->input(4)->node()->kind() == prim::Constant &&
        node->input(7)->node()->kind() == prim::Constant;

    // TODO: more check to make sure

    return has;
  }

  bool Conv2dNode_hasConstantParams(Node* node) const {
    bool has = true;
    for (int i = 1; i < node->inputs().size(); ++i) {
      has = has && (node->input(i)->node()->kind() == prim::Constant);
    }

    return has;
  }

  //
  // currently we only have to fold conv2d + batch_norm
  //
  bool isFoldable(Node* node, Node* prev) {
    bool foldable =
        (node->kind() == aten::batch_norm &&
         (prev->kind() == aten::conv2d || prev->kind() == aten::_convolution));

    //
    // Check whether all the sources are constant ???
    // Does performance improve no matter we do it pre-compiling or runtime?
    //

    foldable = foldable && Conv2dNode_hasConstantParams(prev) &&
        BatchNorm2dNode_hasConstantParams(node);
    return foldable;
  }

  Node* foldNodes(Node* conv2d, Node* batch_norm) {
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

  Node* createBatchNormFoldWeight(Node* conv2d, Node* batch_norm) {
    auto g = conv2d->owningGraph();
    auto newNode = g->create(xpu::fold_weight_sym);
    newNode->setScope(conv2d->scope());

    // We need following parameters
    newNode->addInput(conv2d->input(1)); // Conv2d weights
    newNode->addInput(batch_norm->input(1)); // Batch norm weights
    newNode->addInput(batch_norm->input(4)); // running_var (delta)
    newNode->addInput(batch_norm->input(7)); // eps

    // We get meta and type from conv2d weight value
    newNode->output()->copyMetadata(conv2d->input(1));
    newNode->output()->setType(conv2d->input(1)->type());
    newNode->output()->setDebugName(
        conv2d->input(1)->debugName() + ".bn_folded");

    return newNode;
  }

  Node* createBatchNormFoldBias(Node* conv2d, Node* batch_norm) {
    auto g = conv2d->owningGraph();
    auto newNode = g->create(xpu::fold_bias_sym);
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
    newNode->output()->setDebugName(
        conv2d->input(2)->debugName() + ".bn_folded");

    return newNode;
  }

  bool needCheckUses(Node* node) {
    vector<c10::Symbol>::iterator it = find(
        not_check_uses_ops.begin(), not_check_uses_ops.end(), node->kind());
    if (it == not_check_uses_ops.end()) {
      return true;
    } else {
      return false;
    }
  }

  bool aliasIsSafeForSquashingValue(Node* node, Value* v) {
    bool safe = false;
    auto prev = v->node();
    if (aliasDb_->moveAfterTopologicallyValid(node, prev)) {
      if (!needCheckUses(node) || v->uses().size() == 1 ||
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
      Node* node,
      int v_used_times) {
    auto ops = getAllOperatorsFor(symbol);

    for (auto& op : ops) {
      auto& schema = op->schema();
      if (schema.arguments().size() ==
              prev->inputs().size() + node->inputs().size() - v_used_times &&
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
      TORCH_INTERNAL_ASSERT(r);
      auto rule = *r.value();
      auto& schema =
          matchSchemaForFusion(rule.second, v->node(), node, v->uses().size());
      auto o_schema = node->schema();

      auto pos = v->node()->inputs().size();

      TORCH_INTERNAL_ASSERT(
          schema.arguments().size() ==
          pos + node->inputs().size() - v->uses().size());

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
      } else if (
          isFoldable(node, prev) && aliasIsSafeForSquashingValue(node, v)) {
        pos = foldNodes(prev, node);
        changed = true;
        break;
      }
    }
    return std::make_pair(++pos->iterator(), changed);
  }

  bool isXPUValue(Value* v) {
    // it is not a tensor type
    if (!v->type()->isSubtypeOf(TensorType::get())) {
      return false;
    }

    auto device = v->type()->expectRef<TensorType>().device();

    if (!device) {
      return false; // this tensor has not device info
    }
    return (device->is_xpu() ? true : false);
  }

  bool isXPUNode(Node* node) {
    bool is_xpu = false;
    for (const auto& output : node->outputs()) {
      is_xpu = is_xpu || isXPUValue(output);
      if (is_xpu)
        return true;
    }
    for (const auto& input : node->inputs()) {
      is_xpu = is_xpu || isXPUValue(input);
      if (is_xpu)
        return true;
    }
    return false;
  }

  bool hasXPUNodes() {
    torch::jit::DepthFirstGraphNodeIterator it(graph_);
    for (auto* node = it.next(); node != nullptr; node = it.next()) {
      if (isXPUNode(node))
        return true;
    }
    return false;
  }
};

// TODO: These rules should be more scalable
OpFuser::RuleTab OpFuser::dnnlRules = {
    // RN50/RCAN: conv + add + relu
    {{xpu::conv2d_sum_sym, aten::relu}, xpu::conv2d_sum_relu_sym},
    {{xpu::conv2d_sum_sym, Symbol::fromQualString("aten::relu_")},
     xpu::conv2d_sum_relu_sym},
    {{xpu::_convolution_sum_sym, aten::relu}, xpu::_convolution_sum_relu_sym},
    {{xpu::_convolution_sum_sym, Symbol::fromQualString("aten::relu_")},
     xpu::_convolution_sum_relu_sym},
    // RN50/RCAN: conv + add
    // note: sum post op can only used in inplace scenario
    // {{aten::conv2d, aten::add}, xpu::conv2d_sum_sym},
    {{aten::conv2d, aten::add_}, xpu::conv2d_sum_sym},
    // RCAN: mul + add
    {{aten::mul, aten::add_}, xpu::mul_add_sym},
    // RN50/RCAN INT8: conv + add + relu
    {{Symbol::fromQualString("quantized::conv2d"),
      Symbol::fromQualString("quantized::add_relu")},
     xpu::q_conv2d_sum_relu_sym},
    // SSD-MobileBet: pad + conv
    {{aten::constant_pad_nd, aten::conv2d}, xpu::pad_conv2d_sym},
    // SSD-MobileNet INT8: conv + leaky_relu_
    {{Symbol::fromQualString("quantized::conv2d"),
      Symbol::fromQualString("aten::leaky_relu")},
     xpu::q_conv2d_leaky_relu_sym},
    {{Symbol::fromQualString("quantized::conv2d"),
      Symbol::fromQualString("aten::leaky_relu_")},
     xpu::q_conv2d_leaky_relu_sym},
    // SE-ResNeXt INT8: conv + sigmoid
    {{Symbol::fromQualString("quantized::conv2d"),
      Symbol::fromQualString("aten::sigmoid")},
     xpu::q_conv2d_sigmoid_sym},
    // YOLOv4 INT8 For ATS-M: conv2d + dequantize
    {{Symbol::fromQualString("quantized::conv2d"),
      Symbol::fromQualString("aten::dequantize")},
     xpu::q_conv2d_dequantize_sym},
    // YOLOv4 INT8 For ATS-M: softplus + tanh
    {{Symbol::fromQualString("aten::softplus"),
      Symbol::fromQualString("aten::tanh")},
     xpu::softplus_tanh_sym},
    // 'block_permution + copy'-> 'block permution'.
    {{Symbol::fromQualString("aten::permute"),
      Symbol::fromQualString("aten::contiguous")},
     xpu::permute_contiguous_sym},
    // YOLOv4 INT8 For ATS-M: softplus_tanh + mul
    {{xpu::softplus_tanh_sym, aten::mul}, xpu::mish_compound_sym},
    {{xpu::softplus_tanh_sym, Symbol::fromQualString("aten::mul")},
     xpu::mish_compound_sym},
    // YOLOv4 INT8 For ATS-M: q_conv2d_dequantize + mish_compound
    {{xpu::q_conv2d_dequantize_sym, xpu::mish_compound_sym},
     xpu::q_conv2d_dequantize_mish_compound_sym},
    // YOLOv4
    // INT8 For ATS-M: q_conv2d_mish_yolo
    {{xpu::q_conv2d_dequantize_mish_compound_sym,
      Symbol::fromQualString("aten::quantize_per_tensor")},
     xpu::q_conv2d_mish_compound_sym},
    // FP16 For YOLOv4 _convolution_mish_yolo
    {{aten::_convolution, xpu::mish_compound_sym},
     xpu::_convolution_mish_compound_sym},
    // FP32 For YOLOv4 conv2d_mish_yolo
    {{aten::conv2d, xpu::mish_compound_sym}, xpu::conv2d_mish_compound_sym},
    // INT8 For ATS-M: q_conv2d_mish_add_yolo
    {{xpu::q_conv2d_mish_compound_sym,
      Symbol::fromQualString("quantized::add")},
     xpu::q_conv2d_mish_compound_add_sym},
    // FP16 For YOLOv4 _conovlution_mish_add_yolo
    {{xpu::_convolution_mish_compound_sym, aten::add_},
     xpu::_convolution_mish_compound_add_sym},
    {{xpu::_convolution_mish_compound_sym, aten::add},
     xpu::_convolution_mish_compound_add_sym},
    // FP32 For YOLOv4 conv2d_mish_add_yolo
    {{xpu::conv2d_mish_compound_sym, aten::add_},
     xpu::conv2d_mish_compound_add_sym},
    {{xpu::conv2d_mish_compound_sym, aten::add},
     xpu::conv2d_mish_compound_add_sym},
    // BERT: linear with bias + add
    {{aten::linear, aten::add}, xpu::linear_sum_sym},
    // BERT: linear no bias + add standalone bias + add
    {{aten::t, aten::matmul}, xpu::t_matmul_sym},
    {{xpu::t_matmul_sym, aten::add}, xpu::t_matmul_add_sym},
    {{xpu::t_matmul_sym, Symbol::fromQualString("aten::add_")},
     xpu::t_matmul_add_sym},
    {{xpu::t_matmul_add_sym, aten::add}, xpu::t_matmul_add_add_sym},
    {{xpu::t_matmul_add_sym, aten::gelu}, xpu::t_matmul_add_gelu_sym},
    // BERT: matmul(m1, m2.t()) * scalar + add
    {{Symbol::fromQualString("aten::transpose"),
      Symbol::fromQualString("aten::matmul")},
     xpu::trans_matmul_sym},
    {{xpu::trans_matmul_sym, Symbol::fromQualString("aten::div")},
     xpu::trans_matmul_div_sym},
    // matmul(m1, m2) + add (bias or post_sum)
    {{Symbol::fromQualString("aten::matmul"), aten::add_}, xpu::matmul_add_sym},
    {{Symbol::fromQualString("aten::dequantize"), aten::pixel_shuffle},
     xpu::dequant_pixelshuffle_sym},
    {{xpu::dequant_pixelshuffle_sym,
      Symbol::fromQualString("aten::quantize_per_tensor")},
     xpu::dequant_pixelshuffle_quant_sym},
    // Note: when model adopting fp16 as datatype and generate modelscript by
    // jit.trace, convolution will be parsed as _convolution on ir.

    // YOLOv4 fp16: conv + add_
    {{Symbol::fromQualString("aten::_convolution"), aten::add_},
     xpu::_convolution_sum_sym},
    {{Symbol::fromQualString("aten::_convolution"), aten::add},
     xpu::_convolution_sum_sym},
    {{xpu::_convolution_sum_sym, Symbol::fromQualString("aten::relu_")},
     xpu::_convolution_sum_relu_sym},
    {{Symbol::fromQualString("aten::_convolution"),
      Symbol::fromQualString("aten::silu_")},
     xpu::convolution_silu_sym},
    {{Symbol::fromQualString("aten::conv2d"),
      Symbol::fromQualString("aten::mul")},
     xpu::conv2d_binary_mul_sym},
    {{Symbol::fromQualString("quantized::cat"),
      Symbol::fromQualString("aten::dequantize")},
     xpu::q_cat_dequantize_sym},

    IPEX_DEFINE_CONV_FUSION(sqrt),
    IPEX_DEFINE_CONV_FUSION(square),
    IPEX_DEFINE_CONV_FUSION(abs),
    IPEX_DEFINE_CONV_FUSION(exp),
    IPEX_DEFINE_CONV_FUSION(log),
    IPEX_DEFINE_CONV_FUSION(round),
    IPEX_DEFINE_CONV_FUSION(silu),
    IPEX_DEFINE_CONV_FUSION(gelu),
    IPEX_DEFINE_CONV_FUSION(log_sigmoid),
    IPEX_DEFINE_CONV_FUSION(hardswish),
    IPEX_DEFINE_CONV_FUSION(mish),
    IPEX_DEFINE_CONV_FUSION(hardsigmoid),
    IPEX_DEFINE_CONV_FUSION(tanh),
    IPEX_DEFINE_CONV_FUSION(leaky_relu),
    IPEX_DEFINE_CONV_FUSION(pow),
    IPEX_DEFINE_CONV_FUSION(elu),
    IPEX_DEFINE_CONV_FUSION(hardtanh),
    IPEX_DEFINE_CONV_FUSION(sigmoid),
    IPEX_DEFINE_CONV_FUSION(leaky_relu),
    IPEX_DEFINE_CONV_FUSION(pow),
    IPEX_DEFINE_CONV_FUSION(relu),
    // IPEX_DEFINE_CONV_FUSION(soft_relu),

    // define linear related fusion pattern
    IPEX_DEFINE_LINEAR_FUSION(sigmoid),
    IPEX_DEFINE_LINEAR_FUSION(relu),
    IPEX_DEFINE_LINEAR_FUSION(sqrt),
    IPEX_DEFINE_LINEAR_FUSION(square),
    IPEX_DEFINE_LINEAR_FUSION(abs),
    IPEX_DEFINE_LINEAR_FUSION(exp),
    IPEX_DEFINE_LINEAR_FUSION(log),
    IPEX_DEFINE_LINEAR_FUSION(round),
    IPEX_DEFINE_LINEAR_FUSION(silu),
    IPEX_DEFINE_LINEAR_FUSION(gelu),
    IPEX_DEFINE_LINEAR_FUSION(log_sigmoid),
    IPEX_DEFINE_LINEAR_FUSION(hardswish),
    IPEX_DEFINE_LINEAR_FUSION(hardsigmoid),
    // Note: linear + mish and linear + tanh fusion will be enabled after oneDNN
    // fix its accuracy issue.
    // IPEX_DEFINE_LINEAR_FUSION(mish),
    // IPEX_DEFINE_LINEAR_FUSION(tanh),
    IPEX_DEFINE_LINEAR_FUSION(leaky_relu),
    IPEX_DEFINE_LINEAR_FUSION(pow),
    IPEX_DEFINE_LINEAR_FUSION(elu),
    IPEX_DEFINE_LINEAR_FUSION(hardtanh),

};

void FusionPass(std::shared_ptr<Graph>& graph) {
  // Pattern based fusion was lack of alias analysis
  // ??? It may either be too conservative or too aggressive ???
  // getSubgraphRewriter().runOnGraph(graph);

  auto xpu_fuser = OpFuser(graph->block(), graph);
  if (!torch::jit::getProfilingMode()) {
    // Case 1: Profiling mode is off, no device info can be touched,
    // eanble fusion with warning.
    TORCH_WARN(
        "IPEX XPU dedicated fusion passes are enabled in ScriptGraph non profiling execution mode. "
        "Please enable profiling execution mode to retrieve device guard. \n");
    OpFuser(graph->block(), graph).run();
  } else if (xpu_fuser.hasXPUNodes()) {
    // Case 2: Profiling mode is on and XPU node exists in graph. Run fusion.
    OpFuser(graph->block(), graph).run();
  } else {
    // Case 3: Profile is on, but no XPU node.
    // Do nothing, since fusion is dangerous
    return;
  }

  // TODO: Some post processing?? ECS/EDC/Peephole???
  ConstantPropagation(graph);
}

} // namespace xpu

RegisterPreFusionPass::RegisterPreFusionPass(GraphPass p) {
  registerPrePass(std::move(p));
}

static RegisterPreFusionPass pass_3([](std::shared_ptr<Graph>& g) {
  RemoveProfileNodesAndSpecializeTypes(g);
  xpu::FusionPass(g);
  // RemoveTensorTypeSpecializations(g);
});

} // namespace jit
} // namespace torch
