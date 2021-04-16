#include "torch_ipex/csrc/LlgaTensorImpl.h"
#include "jit/codegen/onednn/graph_helper.h"
#include "jit/codegen/onednn/fusion_group_name.h"

#include <ATen/core/functional.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

using opkind = dnnl::graph::op::kind;

struct DequantInfo {
  size_t unique_id;
  std::vector<float> scales;
  std::vector<int64_t> zps;
  std::string qtype;
  std::string in_type;
  int64_t axis;
};

static std::unordered_map<size_t, DequantInfo> DequantMap;

dnnl::graph::logical_tensor::data_type getQuantizationDataType(std::string dt) {
  
    if (dt == std::string("int8")) {
      return dnnl::graph::logical_tensor::data_type::s8;
    } else if (dt == std::string("uint8")) {
      return dnnl::graph::logical_tensor::data_type::u8;
    } else {
      TORCH_CHECK(false, "Incorrect Quantization data type ", dt);
    }
}

void fixConvOptionalBias(Node* node) {
  if (!node->input(2)->mustNotBeNone()) {
    // Replace non-existent optional bias with const None
    auto g = node->owningGraph();
    auto n = g->createNone();
    auto v = n->insertBefore(node)->output();
    node->replaceInput(2, v);
  }
}

c10::optional<size_t> getDimensions(Value* v) {
  if (v->type()->isSubtypeOf(TensorType::get()))
    return v->type()->cast<TensorType>()->sizes().size();
  else
    return c10::nullopt;
}

Operator makeWildcardOp(Node* node) {
  auto o = Operator(node, opkind::Wildcard);
  // wildcard op contains only topology info
  for (size_t i = 0; i < node->inputs().size(); i++)
    o.setInput(i);
  for (size_t i = 0; i < node->outputs().size(); i++)
    o.setOutput(i);
  return o;
}

#define REQ(cond)                                     \
  if (!(cond)) {                                      \
    GRAPH_DEBUG("Unsupported condition " #cond "\n"); \
    return makeWildcardOp(node);                      \
  }

Operator makeEltwiseOp(Node* node, opkind kind) {
  return Operator(node, kind).setInput(0).setOutput(0);
}

Operator makeBinaryOp(Node* node, opkind kind) {
  REQ(node->input(0)->type()->isSubtypeOf(TensorType::get()) &&
      node->input(1)->type()->isSubtypeOf(TensorType::get()))
  return Operator(node, kind).setInput(0, 1).setOutput(0);
}

// For dequantize, the zp and scale is found through the input node which is 
// a quantize_per_tensor or a quantize_per_channel node.
// Not able to get it directly from the input tensor during compile time
Operator makeDequantOp(Node* node, Node* input_node) {
  if (input_node->kind() == Symbol::aten("quantize_per_tensor")) {
    DequantInfo info = {
      node->output(0)->unique(), 
      Operator::FloatToVector(input_node, 1), 
      Operator::IntToVector(input_node, 2), 
      std::string("per_tensor"),
      Operator::String(input_node, 3),
      -1
    };
    DequantMap.insert(std::pair<size_t, DequantInfo>(node->output(0)->unique(), info));

    // TODO: change the attr name to unique_id
    node->i_(Symbol::attr("dequant_id"), node->output(0)->unique());

    return Operator(node, opkind::Dequantize)
      .setQuantizationInputValue(node->input(0), getQuantizationDataType(Operator::String(input_node, 3)))
      .setOutput(0)
      .setAttr("scales", Operator::FloatToVector(input_node, 1))
      .setAttr("zps", Operator::IntToVector(input_node, 2))
      .setAttr("in_type", Operator::String(input_node, 3))
      .setAttr("qtype", std::string("per_tensor"));
  }
  else {
    DequantInfo info = {
      node->output(0)->unique(), 
      Operator::FloatTensorToVector(input_node, 1), 
      Operator::IntTensorToVector(input_node, 2), 
      std::string("per_channel"),
      Operator::String(input_node, 4),
      Operator::Int(input_node, 3)
    };
    DequantMap.insert(std::pair<size_t, DequantInfo>(node->output(0)->unique(), info));

    // TODO: change the attr name to unique_id
    node->i_(Symbol::attr("dequant_id"), node->output(0)->unique());
    return Operator(node, opkind::Dequantize)
      .setQuantizationInputValue(node->input(0), getQuantizationDataType(Operator::String(input_node, 4)))
      .setOutput(0)
      .setAttr("scales", Operator::FloatTensorToVector(input_node, 1))
      .setAttr("zps", Operator::IntTensorToVector(input_node, 2))
      .setAttr("axis", Operator::Int(input_node, 3))
      .setAttr("in_type", Operator::String(input_node, 4))
      .setAttr("qtype", std::string("per_channel"));
  }
}

Operator createOperator(Node* node) {
  // switch does not allow non-constexpr function, to make the Symbol constexpr, 
  // we must add them to the list in aten/src/ATen/core/interned_strings.h to explicitly use interned strings as symbols.
  // Thus, we use if-else here instead of switch to avoid having to apply patch on PyTorch.
  if (node->kind() == Symbol::aten("conv2d")) {
    fixConvOptionalBias(node);
    return Operator(node, opkind::Convolution)
        .setInput(0, 1, 2)
        .setOutput(0)
        .setAttr("strides", Operator::Ints, 3)
        .setAttr("pads_begin", Operator::Ints, 4)
        .setAttr("pads_end", Operator::Ints, 4)
        .setAttr("dilations", Operator::Ints, 5)
        .setAttr("groups", Operator::Int, 6)
        .setAttr("filter_format", std::string("OIX"));
  } else if (node->kind() == Symbol::aten("_convolution")) {
    bool transposed = Operator::Bool(node, 6);
    REQ(!transposed);

    return Operator(node, opkind::Convolution)
        .setInput(0, 1, 2)
        .setOutput(0)
        .setAttr("strides", Operator::Ints, 3)
        .setAttr("pads_begin", Operator::Ints, 4)
        .setAttr("pads_end", Operator::Ints, 4)
        .setAttr("dilations", Operator::Ints, 5)
        .setAttr("groups", Operator::Int, 8)
        .setAttr("filter_format", std::string("OIX"));
  } else if (node->kind() == Symbol::aten("batch_norm")) {
    auto training = toIValue(node->input(5));
    REQ(training.has_value()); // cannot get training status in script mode
    REQ(!training->toBool()); // TODO: support bn training
    return Operator(node, opkind::BatchNormInference)
        .setInput(0, 1, 2, 3, 4)
        .setOutput(0)
        .setAttr("epsilon", Operator::Float, 7);
  } else if (node->kind() == Symbol::aten("layer_norm")) {
    auto normalized_shape = Operator::Ints(node, 1);
    REQ(normalized_shape.size() == 1);
    return Operator(node, opkind::LayerNorm)
        .setInput(0, 2, 3)
        .setOutput(0)
        .setAttr("epsilon", Operator::Float, 4)
        .setAttr("keep_stats", false);
  } else if (node->kind() == Symbol::aten("add")) {
    return makeBinaryOp(node, opkind::Add);
  } else if (node->kind() == Symbol::aten("tanh")) {
    return makeEltwiseOp(node, opkind::Tanh);
  } else if (node->kind() == Symbol::aten("relu")) {
    return makeEltwiseOp(node, opkind::ReLU);
  } else if (node->kind() == Symbol::aten("elu")) {
    return makeEltwiseOp(node, opkind::Elu)
        .setAttr("alpha", Operator::Float, 1);
  } else if (node->kind() == Symbol::aten("sigmoid")) {
    return makeEltwiseOp(node, opkind::Sigmoid);
  } else if (node->kind() == Symbol::aten("gelu")) {
    return makeEltwiseOp(node, opkind::GELU);
  } else if (node->kind() == Symbol::aten("sqrt")) {
    return makeEltwiseOp(node, opkind::Sqrt);
  } else if (node->kind() == Symbol::aten("abs")) {
    return makeEltwiseOp(node, opkind::Abs);
  } else if (node->kind() == Symbol::aten("square")) {
    return makeEltwiseOp(node, opkind::Square);
  } else if (node->kind() == Symbol::aten("hardtanh")) {
    return makeEltwiseOp(node, opkind::HardTanh)
        .setAttr("min", Operator::Float, 1)
        .setAttr("max", Operator::Float, 2);
  } else if (node->kind() == Symbol::aten("softmax")) {
    auto dim0 = getDimensions(node->input(0));
    REQ(dim0.has_value());

    auto axis = Operator::Int(node, 1);
    if (axis < 0)
      axis += dim0.value();

    return Operator(node, opkind::SoftMax)
        .setInput(0)
        .setOutput(0)
        .setAttr("axis", axis);
  } else if (node->kind() == Symbol::aten("cat")) {
    return makeWildcardOp(node); // TODO: remove once Concat is supported

    auto o = Operator(node, opkind::Concat);
    REQ(node->input(0)->node()->kind() == prim::ListConstruct);
    REQ(node->input(0)->uses().size() == 1);
    REQ(node->input(1)->node()->kind() == prim::Constant);
    // aten::cat needs a special handling since it takes a Tensor[] as input.
    // We set the inputs of ListConstruct as the inputs of cat.
    //
    // Pytorch IR:                              LLGA sees:
    //     %a    %b     %c          %dim              %a    %b    %c
    //      \     |     /             |                \     |    /
    //   prim::ListConstruct   prim::Constant     llga::Concat[axis=%dim]
    //                    \      /
    //                    aten::cat
    auto listConstruct = node->input(0)->node();
    for (auto input : listConstruct->inputs())
      o.setInputValue(input);
    return o.setOutput(0).setAttr("axis", Operator::Int, 1);
  } else if (node->kind() == Symbol::aten("max_pool2d")) {
    // TODO: disable for RN50 int8 path?
    auto rounding_type = Operator::Bool(node, 5) ? "ceil" : "floor";
    return Operator(node, opkind::MaxPool)
        .setInput(0)
        .setOutput(0)
        .setAttr("kernel", Operator::Ints, 1)
        .setAttr("strides", Operator::Ints, 2)
        .setAttr("pads_begin", Operator::Ints, 3)
        .setAttr("pads_end", Operator::Ints, 3)
        .setAttr("dilations", Operator::Ints, 4)
        .setAttr("rounding_type", std::string(rounding_type));
  } else if (node->kind() == Symbol::aten("avg_pool2d")) {
    auto rounding_type = Operator::Bool(node, 4) ? "ceil" : "floor";
    auto divisor_override = toIValue(node->input(6));
    REQ(divisor_override->isNone());
    return Operator(node, opkind::AvgPool)
        .setInput(0)
        .setOutput(0)
        .setAttr("kernel", Operator::Ints, 1)
        .setAttr("strides", Operator::Ints, 2)
        .setAttr("pads_begin", Operator::Ints, 3)
        .setAttr("pads_end", Operator::Ints, 3)
        .setAttr("exclude_pad", !Operator::Bool(node, 5))
        .setAttr("rounding_type", std::string(rounding_type));
  } else if (node->kind() == Symbol::aten("matmul")) {
    auto dim0 = getDimensions(node->input(0)).value_or(-1);
    auto dim1 = getDimensions(node->input(1)).value_or(-1);
    // TODO: support all shape combinations
    // TODO: cannot get dims here after updating PyTorch
    REQ((dim0 == 2 && dim1 == 2) || (dim0 == 4 && dim1 == 4) ||
        (dim0 == 3 && dim1 == 2));
    // fall through
    return Operator(node, opkind::MatMul).setInput(0, 1).setOutput(0);
  } else if (node->kind() == Symbol::aten("mm")) {
    return Operator(node, opkind::MatMul).setInput(0, 1).setOutput(0);
  } else if (node->kind() == Symbol::aten("linear")) {

    // TODO: cannot get dims here after updating PyTorch
    auto dim0 = getDimensions(node->input(0)).value_or(-1);
    auto dim1 = getDimensions(node->input(1)).value_or(-1);
    // REQ(dim1 == 2);

    return Operator(node, opkind::MatMul)
        .setInput(0, 1, 2)
        .setOutput(0)
        .setAttr("transpose_b", true);
  } else if (node->kind() == Symbol::aten("quantize_per_tensor")) {
    // TODO: how to handle this case
    REQ(node->input(1)->node()->kind() != Symbol::aten("q_scale"));
    
    // TODO: how to handle this case:
    //      quantize_per_tensor
    //   ---/-----/-----\-----\---
    //dequant q_scale  q_zp  dtype
    // REQ(node->output(0)->uses().size() <= 2);

    return Operator(node, opkind::Quantize)
        .setInput(0)
        .setQuantizationOutputValue(node->output(0), getQuantizationDataType(Operator::String(node, 3)))
        .setAttr("scales", Operator::FloatToVector, 1)
        .setAttr("zps", Operator::IntToVector, 2)
        .setAttr("out_type", Operator::String, 3)
        .setAttr("qtype", std::string("per_tensor"));
  } else if (node->kind() == Symbol::aten("quantize_per_channel")) {
    return Operator(node, opkind::Quantize)
      .setInput(0)
      .setQuantizationOutputValue(node->output(0), getQuantizationDataType(Operator::String(node, 4)))
      .setAttr("scales", Operator::FloatTensorToVector, 1)
      .setAttr("zps", Operator::IntTensorToVector, 2)
      .setAttr("axis", Operator::Int, 3)
      .setAttr("out_type", Operator::String, 4)
      .setAttr("qtype", std::string("per_channel"));
  } else if (node->kind() == Symbol::aten("dequantize")) {
    if (!node->hasAttribute(Symbol::attr("dequant_id"))) {
      Node* input_node = node->input(0)->node();
      // TODO: how to handle input(1) == Symbol::aten("q_scale")
      REQ(((input_node->kind() == Symbol::aten("quantize_per_tensor")) || (input_node->kind() == Symbol::aten("quantize_per_channel"))) && (input_node->input(1)->node()->kind() != Symbol::aten("q_scale")));

      // TODO: how to handle this case:
      //      quantize_per_tensor
      //   ---/-----/-----\-----\---
      //dequant q_scale  q_zp  dtype
      // REQ(input_node->output(0)->uses().size() <= 2);

      return makeDequantOp(node, input_node);
    } else {
      // save the zp, scale... into the node to retrieve it when the node has been added into LLGA fusion group
      auto unique_id = node->i(Symbol::attr("dequant_id"));
      auto it = DequantMap.find(unique_id);
      
      REQ(it != DequantMap.end());

      DequantInfo dequant_info = it->second;

      if (dequant_info.qtype == std::string("per_tensor")) {
        return Operator(node, opkind::Dequantize)
          .setQuantizationInputValue(node->input(0), getQuantizationDataType(dequant_info.in_type))
          .setOutput(0)
          .setAttr("scales", dequant_info.scales)
          .setAttr("zps", dequant_info.zps)
          .setAttr("in_type", dequant_info.in_type)
          .setAttr("qtype", dequant_info.qtype);
      } else {
        return Operator(node, opkind::Dequantize)
          .setQuantizationInputValue(node->input(0), getQuantizationDataType(dequant_info.in_type))
          .setOutput(0)
          .setAttr("scales", dequant_info.scales)
          .setAttr("zps", dequant_info.zps)
          .setAttr("axis", dequant_info.axis)
          .setAttr("in_type", dequant_info.in_type)
          .setAttr("qtype", dequant_info.qtype);
      }
    }
  }
  return makeWildcardOp(node);
}

dnnl::graph::op createLlgaOp(Node* node) {
  return createOperator(node).llgaOp();
}

bool isSupported(Node* node) {
  // return createOperator(node).kind() != opkind::Wildcard;
  // TODO: special handling here for the below WildCard OPs:
  //   upsample_nearest2d, 
  //   TupleConstruct
  //   size
  // Since the usage of Wildcard OP is still not well-defined, 
  // we have decided previously not to send the Wildcard OPs to LLGA.
  // As a result, for the below pattern, LLGA is not aware of the
  // WildCard OP: TupleConstruct and will wrongly select the 
  // dequant-conv-quant into a partition.
  // Workaround here to send the upsample_nearest2d, TupleConstruct and size
  // to LLGA.
  // In the future, need define the usage of WildCard OPs and send 
  // all of them to LLGA to correctly select the partitions.
  //
  //                 quant
  //      + - - - - - -|- - - - - - +
  //      |        dequant          |
  //      |           |             |
  //      |         conv            |
  //      |       /      \          |
  //      |  quant    TupleConstruct|
  //      + - | - - - - - - | - - - +
  // 
  return node->kind() == aten::upsample_nearest2d || node->kind() == prim::TupleConstruct || node->kind() == aten::size || createOperator(node).kind() != opkind::Wildcard;
};

DeviceType inferDeviceFromValue(Value* v) {
  auto tt = v->type()->cast<TensorType>();
  if (!tt)
    return at::kCPU;
  auto device = tt->device();
  if (!device)
    return at::kCPU;
  return device->type();
}

DeviceType inferDevice(const std::shared_ptr<Graph>& graph) {
  auto dt = inferDeviceFromValue(graph->inputs()[0]);
  TORCH_CHECK(
      std::all_of(
          graph->inputs().begin(),
          graph->inputs().end(),
          [dt](Value* v) { return inferDeviceFromValue(v) == dt; }),
      "All inputs must have the same deive type");
  return dt;
}

dnnl::graph::engine::kind getLlgaEngineKind(DeviceType type) {
  switch (type) {
    case DeviceType::CPU:
      return dnnl::graph::engine::kind::cpu;
    default:
      TORCH_CHECK(false, "Not support device type ", type);
  }
}

void mayAddListConstructIntoConcatPartition(
    Node* n,
    OpPartitionMap& opToOwningPartition) {
  // Since prim::ListConstruct is not visible to the LLGA,
  // it will not be in any partition returned from partfuseritioning results.
  // We need rewrite opToOwningPartition to make the prim::ListConstruct to be
  // virtually in the same partition with the aten::cat, so that
  // prim::ListConstruct can be fused into the fusion group by graph fuser
  if (n->kind() == aten::cat && opToOwningPartition.has(n)) {
    auto listConstrcut = n->input(0)->node();
    auto partitionId = opToOwningPartition.get(n);
    opToOwningPartition.add(listConstrcut, partitionId);
  }
}

LlgaGraphHelper::LlgaGraphHelper(
    const std::shared_ptr<Graph>& graph,
    dnnl::graph::partition::policy policy) {
  auto deviceType = inferDevice(graph);
  auto engineKind = getLlgaEngineKind(deviceType);
  dnnl::graph::graph g{engineKind};

  GRAPH_DEBUG("Constructing LLGA graph");
  // TODO: select nodes in top-level block for now
  for (auto* node : graph->block()->nodes()) {
    // TODO: remove once wildcard is supported
    if (!isSupported(node))
      continue;
    auto op = createLlgaOp(node);
    g.add_op(op);
    GRAPH_DEBUG("  Added node ", node->kind().toQualString());
  }

  GRAPH_DEBUG("Get Partitions");
  partitions = g.get_partitions(policy);

  GRAPH_DEBUG("  Got #partitions: ", partitions.size());
  for (size_t partId = 0; partId < partitions.size(); partId++) {
    if (partitions[partId].is_supported()) {
      for (auto opId : partitions[partId].get_ops()) {
        opToOwningPartition.add(opId, partId);
      }
    }
  }

  // Scanning the graph again for post processing
  for (auto* node : graph->block()->nodes()) {
    mayAddListConstructIntoConcatPartition(node, opToOwningPartition);
  }
}

bool LlgaGraphHelper::isLlgaSubgraph(const Node* node) {
  return node->hasAttribute(attr::Subgraph) &&
      node->kind() == Symbol::fromQualString(LlgaFusionGroupName());
}

bool LlgaGraphHelper::shouldMerge(Node* toMerge, Node* subgraph) {
  TORCH_CHECK(
      isLlgaSubgraph(subgraph),
      "The consumer node does not contain a subgraph");
  if (!shouldConsiderForMerge(toMerge)) {
    return false;
  }
  return opToOwningPartition.get(toMerge) == opToOwningPartition.get(subgraph);
}

bool isViewOp(Node* n) {
  switch (n->kind()) {
    case aten::view:
    case aten::view_as:
    case aten::reshape:
    case aten::reshape_as:
    case aten::transpose:
    case aten::expand:
    case aten::expand_as:
      return true;
  }
  return false;
}

bool LlgaGraphHelper::shouldConsiderForMerge(Node* node) {
  // if we're already in the process of merging
  if (isLlgaSubgraph(node)) {
    return true;
  }
  if (isViewOp(node)) {
    return false;
  }
  return opToOwningPartition.has(node);
}

Node* LlgaGraphHelper::createSingletonSubgraph(Node* n, AliasDb& aliasDb) {
  auto partitionId = opToOwningPartition.get(n);
  GRAPH_DEBUG(
      "Creating FusionGroup_", partitionId, " for ", n->kind().toQualString());
  auto group = SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
      n, Symbol::fromQualString(LlgaFusionGroupName()), aliasDb);
  opToOwningPartition.add(group, partitionId);
  LlgaNodeWrapper(group).initOutputLayouts();
  LlgaNodeWrapper(group).initOutputDtypes();
  return group;
}

void LlgaGraphHelper::mergeNodeIntoSubgraph(
    Node* toMerge,
    Node* subgraphNode,
    AliasDb& aliasDb) {
  if (isLlgaSubgraph(toMerge)) {
    GRAPH_DEBUG(
        "Merging ",
        toMerge->kind().toQualString(),
        "_",
        opToOwningPartition.get(toMerge),
        " into ",
        subgraphNode->kind().toQualString(),
        "_",
        opToOwningPartition.get(subgraphNode));
  } else {
    GRAPH_DEBUG(
        "Merging ",
        toMerge->kind().toQualString(),
        " into ",
        subgraphNode->kind().toQualString(),
        "_",
        opToOwningPartition.get(subgraphNode));
  }

  SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
      toMerge, subgraphNode, aliasDb);
}

void LlgaGraphHelper::unmergeIfAnyNodeIsMissing(Node* subgraphNode) {
  TORCH_CHECK(isLlgaSubgraph(subgraphNode), "Cannot unmerge a non-LLGA node");

  auto partitionId = opToOwningPartition.get(subgraphNode);
  auto expectOpNum = partitions[partitionId].get_ops_num();
  auto actualOpNum = countSupportedOps(subgraphNode->g(attr::Subgraph));

  if (expectOpNum != actualOpNum) {
    GRAPH_DEBUG(
        "Unmerging FusionGroup_",
        partitionId,
        ". Expected ",
        expectOpNum,
        " ops, but got ",
        actualOpNum,
        " ops.");
    SubgraphUtils::unmergeSubgraph(subgraphNode);
  }
}

size_t LlgaGraphHelper::countSupportedOps(
    const std::shared_ptr<Graph>& graph) const {
  // TODO: count nodes in top-level block for now
  
  // TODO: This check happends when the node is already in the partition. We assume that is this case 
  // the dequantize node is supported since it has been selected before.
  // We cannot use the isSupported on dequant node since the node before has been rewrite into another partition.
  // One possible solution is to save the zp, scale, axis... of dequant on the node iteself. 
  size_t cnt = 0;
  for (auto* node : graph->block()->nodes())
    if (isSupported(node))
      cnt++;
  return cnt;
}

bool LlgaGraphHelper::shouldSkipEliminateCommonSubexpression(
  const std::shared_ptr<Graph>& graph) {
  // TODO: check nodes in top-level block for now
  for (auto* node : graph->block()->nodes()) {
    if (node->kind() == Symbol::aten("quantize_per_tensor") || node->kind() == Symbol::aten("quantize_per_channel")) {
      if (node->output(0)->uses().size() > 1) {
        for (int i = 0; i < node->output(0)->uses().size(); i++) {
          if (node->output(0)->uses()[i].user->kind() != Symbol::aten("dequantize")) {
            return false;
          }
          return true;
        }
      }
    }
  }
  return false;
}

std::vector<dnnl::graph::partition> LlgaGraphHelper::getPartitions() const {
  return partitions;
}

LlgaNodeWrapper::LlgaNodeWrapper(const Node* node)
    : n(const_cast<Node*>(node)) {
  TORCH_CHECK(
      LlgaGraphHelper::isLlgaSubgraph(n), "Cannot wrap a non-LLGA fusion node");
}

void LlgaNodeWrapper::setOpaqueLayout(size_t offset) {
  TORCH_CHECK(offset < n->outputs().size(), "Invalid output offset ", offset);
  auto& layouts =
      const_cast<std::vector<int64_t>&>(n->is(Symbol::attr("output_layouts")));
  layouts.at(offset) = 1;
}

bool LlgaNodeWrapper::useOpaqueLayout(size_t offset) const {
  TORCH_CHECK(offset < n->outputs().size(), "Invalid output offset ", offset);
  return n->is(Symbol::attr("output_layouts"))[offset] == 1;
}

void LlgaNodeWrapper::initOutputLayouts() {
  if (n->hasAttribute(Symbol::attr("output_layouts"))) {
    return;
  }

  // Init all output layouts as undef
  std::vector<int64_t> layouts(n->outputs().size(), 0);
  n->is_(Symbol::attr("output_layouts"), layouts);
}

void LlgaNodeWrapper::setOutputDtypes(size_t offset, int64_t dtype) {
  TORCH_CHECK(offset < n->outputs().size(), "Invalid output offset ", offset);
  auto& layouts =
      const_cast<std::vector<int64_t>&>(n->is(Symbol::attr("output_dtypes")));
  layouts.at(offset) = dtype;
}

int64_t LlgaNodeWrapper::getOutputDtypes(size_t offset) const {
  TORCH_CHECK(offset < n->outputs().size(), "Invalid output offset ", offset);
  return n->is(Symbol::attr("output_dtypes"))[offset];
}

void LlgaNodeWrapper::initOutputDtypes() {
  if (n->hasAttribute(Symbol::attr("output_dtypes"))) {
    return;
  }

  // Init all output dtypes as undef
  std::vector<int64_t> dtypes(n->outputs().size(), 0);
  n->is_(Symbol::attr("output_dtypes"), dtypes);
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch