#include "fusion_pass.h"
#include <string>
#include "codegen/onednn/interface.h"
#include "cpu/kernels/Matmul.h"
#include "cpu/passes/concat_linear.h"
#include "cpu/passes/frozen_conv_folding.h"
#include "cpu/passes/frozen_linear_folding.h"
#include "cpu/passes/graph_rewrite.h"
#include "cpu/passes/graph_rewrite_helper.h"
#include "cpu/passes/prepack_folding.h"
#include "cpu/passes/remove_redundant_aliases.h"

#include <c10/util/hash.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/batch_mm.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/operator.h>

// XXX: move to somewhere convenient
namespace std {
template <>
struct hash<std::pair<torch::jit::Symbol, torch::jit::Symbol>> {
  size_t operator()(
      std::pair<torch::jit::Symbol, torch::jit::Symbol> pair) const {
    return std::hash<uint64_t>()(
        static_cast<uint64_t>(pair.first) << 32 |
        static_cast<uint64_t>(pair.second));
  }
};
} // namespace std

namespace torch_ipex {
namespace jit {

using namespace torch::jit;

// Including in-place optimizations that try to (conditionally)
// replace the origin op with in-place opted one for better performance.
// This in-place optimized ops may come from either oneDNN or aten
void ApplyInplaceOptimization(std::shared_ptr<Graph>& graph) {
  // try to replace aten ops with ipex in-place ops
  graph_rewrite::replaceAtenOpsWithIpexInplaceOps(graph);
  // try to replace aten ops with aten in-place ops
  graph_rewrite::replaceOpsWithAtenInplaceOps(graph);
}

void ReplaceInplaceOpsWitOutplaceOps(std::shared_ptr<Graph>& graph) {
  graph_rewrite::replaceInplaceOpsWithOutplaceOps(graph);
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

class ATenLinearRecorder {
 public:
  ATenLinearRecorder(std::shared_ptr<Graph> graph) {
    graph_rewrite::RecordAtenLinearNodes(
        graph, aten_linear_nodes_, use_mkl_sgemm);
  }

  std::unordered_set<Node*>& get_records() {
    return aten_linear_nodes_;
  }

  bool& use_mkl() {
    return use_mkl_sgemm;
  }

 private:
  std::unordered_set<Node*> aten_linear_nodes_;
  bool use_mkl_sgemm = false;
};

void IPEXFusionPass(std::shared_ptr<Graph>& graph) {
  // remove dropout;
  torch::jit::removeDropout(graph);

  // ipex einsum
  graph_rewrite::FusedEinsumPost(graph);

  // Fuse the scores calculation(dim + matmul + (add)? + softmax) for
  // Multi-Head-Attention
  graph_rewrite::FuseMHAScoreCalc(graph);

  // Fuse bmm + add for bmm_add
  graph_rewrite::fuseBmmAdd(graph);

  // Replace _convolution with conv2d or conv3d
  torch_ipex::jit::graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  // Replace torch_ipex::convolution_forward with conv2d or conv3d when conv
  // weights are constant. Conv weights will be unpacked in this step.
  graph_rewrite::replaceFrozenIPEXConvWithAtenConv(graph);

  // convolution folding
  graph_rewrite::FrozenConvFolding(graph);

  // Insert ipex_prepack::convolution_prepack.
  // Conv weights will be re-prepacked in this step.
  GRAPH_DUMP("After FrozenConvFolding.Before insertPrePackedConvOp", graph);
  graph_rewrite::insertPrePackedConvOp(graph);

  // convolution fusion
  GRAPH_DUMP("After insertPrePackedConvOp.Before fuseConvWithEltwise", graph);
  graph_rewrite::fuseConvWithEltwise(graph);
  GRAPH_DUMP("After fuseConvWithEltwise.Before fuseConvAddRelu", graph);
  graph_rewrite::fuseConvAddRelu(graph);
  GRAPH_DUMP("After fuseConvAddRelu.Before fuseBottleneck", graph);
  graph_rewrite::fuseBottleneck(graph);
  GRAPH_DUMP("After fuseBottleneck.", graph);

  // TODO: Record original aten nodes, while convert aten linear-> ipex linear,
  // will ignore these aten linear (if they are fp32 dtype). For BF16 dtype,
  // always use ipex linear. This is a temporay solution, for next PR to clean
  // up fusion pass, will further abstract this as a class method.
  auto aten_linear_recorder = ATenLinearRecorder(graph);
  // linear folding
  graph_rewrite::replaceFrozenIPEXLinearWithAtenLinear(
      graph, aten_linear_recorder.use_mkl());
  // concat multi-linear with same input
  torch_ipex::jit::FrozenConcatLinear(
      graph, aten_linear_recorder.get_records());
  graph_rewrite::FrozenLinearFolding(graph);

  // linear fusion
  GRAPH_DUMP("After FrozenLinearFolding.Before insertPrePackedLinearOp", graph);
  graph_rewrite::insertPrePackedLinearOp(
      graph,
      aten_linear_recorder.get_records(),
      aten_linear_recorder.use_mkl());
  GRAPH_DUMP(
      "After insertPrePackedLinearOp.Before fuseLinearWithEltwise", graph);
  graph_rewrite::fuseLinearWithEltwise(graph);
  GRAPH_DUMP("After fuseLinearWithEltwise.Before fuseLinearAddRelu", graph);
  graph_rewrite::fuseLinearAddRelu(graph);
  GRAPH_DUMP("After fuseLinearAddRelu.", graph);

  GRAPH_DEBUG("Before replacing hardsigmoid", graph);
  graph_rewrite::ReplaceHardsigmoidWithIPEX(graph);
  GRAPH_DEBUG("After replacing hardsigmoid", graph);

  graph_rewrite::FuseLinearSwishCustomized(graph);

  // fuse add+layernorm
  graph_rewrite::FuseAddLayerNorm(graph);

  // deconvolution fusion
  GRAPH_DUMP(
      "After FuseAddLayerNorm.Before insertPrePackedConvTransposeOp", graph);
  graph_rewrite::insertPrePackedConvTransposeOp(graph);
  GRAPH_DUMP(
      "After insertPrePackedConvTransposeOp.Before fuseConvTransposeWithEltwise",
      graph);
  graph_rewrite::fuseConvTransposeWithEltwise(graph);
  GRAPH_DUMP(
      "After fuseConvTransposeWithEltwise.Before fuseConvTransposeAdd", graph);
  graph_rewrite::fuseConvTransposeAdd(graph);
  GRAPH_DUMP("After fuseConvTransposeAdd.", graph);

  // fuse concat+bn+relu for the input float tensors with the same sizes
  // and channelslast format
  // hence the concat dim should be the channel
  graph_rewrite::FuseConcatBnRelu(graph);

  // Fuse operators as shuffle
  graph_rewrite::FuseShuffle(graph);
  graph_rewrite::FuseMatmulDivOrMul(graph);

  // replace aten max_pool2d with ipex max_pool2d
  graph_rewrite::replaceAtenMaxPool2dWithIpexMaxPool2d(graph);

  // replace aten softmax with ipex softmax
  graph_rewrite::replaceAtenSoftmaxWithIpexSoftmax(graph);

  // replace aten::batch_norm with ipex::batch_norm, it will be removed
  // after TensorExprs fix the performance issue(IPB-808).
  graph_rewrite::replaceAtenBatchNormWithIpexBatchNorm(graph);
  // TODO: Some post processing?? ECS/EDC/Peephole???

  // This path contains two functions:
  // 1. Fuse BF16 Mha for BERT and ViT
  // 2. Replace the Matmul OP with MKL or DNNL Matmul kernels to enable
  // transpose-free FP32 BMM.
  // The BF16 Mha fusions depends on the FuseMHAScoreCalc and
  // FuseMatmulDivOrMul since it uses the fused OPs from the two pathes.
  // TODO: We will enable transpose-free BF16 BMM to benefit for most
  // of the MHA patterns. Then the Matmul OP from the FuseMHAScoreCalc
  // path will be removed in the future for simplicity.
  // This path should be executed after all the other Matmul-related
  // fusion are completed to prevent mismatching "aten::matmul".
  graph_rewrite::FusedTransFreeMha(graph);

  ConstantPropagation(graph);
  GRAPH_DUMP("Before PrePackingOpsFolder", graph);
  // folding prepacking ops.
  PrePackingOpsFolder(graph);
  GRAPH_DUMP("After PrePackingOpsFolder", graph);
}

bool checkQuantization(Block* block) {
  for (auto node : block->nodes()) {
    for (auto sub : node->blocks()) {
      checkQuantization(sub);
    }

    if (node->kind() == Symbol::aten("quantize_per_tensor") ||
        node->kind() == Symbol::aten("dequantize") ||
        node->kind() == Symbol::aten("quantize_per_channel") ||
        node->kind() == Symbol::aten("quantized_lstm") ||
        node->kind() == Symbol::fromQualString("quantized::linear_dynamic")) {
      return true;
    }
  }
  return false;
}

bool isQuantized(const std::shared_ptr<Graph>& graph) {
  return checkQuantization(graph->block());
}

FusionBehavior getCurrentBehavior(size_t remaining_depth) {
  size_t curr_depth = 0;
  FusionStrategy fusion_strategy_ = getFusionStrategy();
  for (int i = static_cast<int>(fusion_strategy_.size()) - 1; i >= 0; i--) {
    curr_depth += fusion_strategy_[i].second;
    if (remaining_depth <= curr_depth) {
      return fusion_strategy_[i].first;
    }
  }
  // should never get here
  TORCH_WARN("Stratgy changed mid-invocation, NYI");
  return FusionBehavior::STATIC;
}

size_t getInstantiatedBailoutDepth() {
  // Initialize bailout_depth from command-line flag.
  size_t depth = 0;
  FusionStrategy fusion_strategy_ = getFusionStrategy();
  for (const auto& pair : fusion_strategy_) {
    depth += pair.second;
  }
  return depth;
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

  if (isQuantized(graph) || fuser::onednn::is_llga_fp32_bf16_enabled()) {
    RemoveRedundantAliases(graph);
    fuser::onednn::fuseGraph(graph);
  }
  GRAPH_DUMP(
      "After LLGA fusion pass. Before ReplaceInplaceOpsWitOutplaceOps", graph);

  // Replace inplace ops with outplace ops so that we do not need to enumerate
  // all inplace and outplace fusion patterns. This replacement is the inverse
  // of ApplyInplaceOptimization. All the replaced ops failed to be fused will
  // be reverted by ApplyInplaceOptimization.
  ReplaceInplaceOpsWitOutplaceOps(graph);
  GRAPH_DUMP(
      "After ReplaceInplaceOpsWitOutplaceOps. Before IPEXFusionPass", graph);

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
    graph_rewrite::replaceAtenToWithIPEXTo(graph);
    auto min_size = getFusionGroupInlining() ? 2 : 1;
    // Here we always get the first valid behavior per the global fusion
    // strategies configured by PyTorch (`getInstantiatedBailoutDepth` always
    // returns the maximum configured depth). This is because IPEX TE fusion is
    // only called the first time of the compilation while the later
    // re-compilations are triggered from inside PyTorch.
    bool dyn_shapes = getCurrentBehavior(getInstantiatedBailoutDepth()) ==
        FusionBehavior::DYNAMIC;
    FuseTensorExprs(graph, min_size, /* composed op*/ false, dyn_shapes);
    graph_rewrite::replaceIPEXToWithAtenTo(graph);
  }

  // Apply IPEX inplace optimization/replacement
  // Note: Since TE is with priority and it has not supported inplace op yet,
  //       we make inplace optimization after TE.
  ApplyInplaceOptimization(graph);
  RemoveTensorTypeSpecializations(graph);
  GRAPH_DUMP(
      "After RemoveTensorTypeSpecializations. End of optimization pass", graph);
}

} // namespace jit
} // namespace torch_ipex
