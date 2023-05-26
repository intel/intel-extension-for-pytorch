#include "interface.h"
#include <oneapi/dnnl/dnnl_graph.hpp>
#include "defer_size_check.h"
#include "fusion_group_name.h"
#include "graph_fuser.h"
#include "guard_shape.h"
#include "kernel.h"
#include "layout_propagation.h"
#include "lift_up_quant.h"
#include "prepare_binary.h"
#include "prepare_dequant.h"
#include "prepare_silu.h"
#include "process_cast.h"
#include "quantization_patterns.h"
#include "remove_mutation.h"

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/operator_options.h>

namespace torch_ipex {
namespace jit {
namespace fuser {
namespace onednn {

using namespace torch::jit;
namespace {
thread_local bool llga_fp32_bf16_enabled = false;
}

bool is_llga_fp32_bf16_enabled() {
  return llga_fp32_bf16_enabled;
}
void set_llga_fp32_bf16_enabled(bool new_enabled) {
  llga_fp32_bf16_enabled = new_enabled;
}

void fuseGraph(std::shared_ptr<Graph>& g) {
  // Follow the process of the tensorexpr_fuser in profiling mode:
  // Remove prim::profile nodes and embed the profile info directly in the
  // IR in value types to avoid breaking the fusion patterns.
  // Will add shape guard after LLGA optimization passes and
  // wipe the tensor type information from the IR, so that it's not
  // accidentally used by any other pass.

  // We rely on the shape specialization and shape guard to ensure the validity
  // of the cached compilation in the kernel, thus only support profiling mode.
  // TODO: add check on LlgaFusionGroup to ensure allShapesAreKnown on nodes to
  // fuse: torch/csrc/jit/passes/tensorexpr_fuser.cpp: allShapesAreKnown
  if (getProfilingMode()) {
    GRAPH_DUMP(
        "Before mutation removal. Beginning of INT8 "
        "optimization pass",
        g);
    IPEXRemoveTensorMutation(g);
    RemoveListMutation(g);
    GRAPH_DUMP("After mutation removal. Before DecomposeOps", g);
    DecomposeOps(g);
    GRAPH_DUMP("After DecomposeOps. Before PrepareBinaryForLLGA", g);
    PrepareBinaryForLLGA(g);
    GRAPH_DUMP("After PrepareBinaryForLLGA. Before PrepareSiluForLLGA", g);
    PrepareSiluForLLGA(g);
    GRAPH_DUMP(
        "After PrepareSiluForLLGA. Before EliminateCommonSubexpression", g);
    EliminateCommonSubexpression(g);
    GRAPH_DUMP(
        "After EliminateCommonSubexpression. Before SaveDequantInformation", g);
    // SaveDequantInformation must be placed before LiftUpQuant
    SaveDequantInformation(g);
    GRAPH_DUMP("After SaveDequantInformation. Before PrepareDequantForLLGA", g);
    // PrepareDequantForLLGA must be placed after EliminateCommonSubexpression
    PrepareDequantForLLGA(g);
    GRAPH_DUMP("After PrepareDequantForLLGA. Before LiftUpQuant", g);
    // LiftUpQuant must be place before DeferSizeCheck
    LiftUpQuant(g);
    GRAPH_DUMP("After LiftUpQuant. Before ProcessCast", g);
    ProcessCast(g);
    GRAPH_DUMP("After ProcessCast. Before DeferSizeCheck", g);
    DeferSizeCheck(g);
    GRAPH_DUMP("After DeferSizeCheck. Before CreateLlgaSubgraphs", g);
    // CreateLlgaSubgraphs must be placed after all the preparation passes above
    CreateLlgaSubgraphs(g);
    GRAPH_DUMP("After CreateLlgaSubgraphs. Before PropagateLayout", g);
    // PropagateLayout must be placed after CreateLlgaSubgraphs
    PropagateLayout(g);
    GRAPH_DUMP("After PropagateLayout. Before RevertPrepareBinaryForLLGA", g);
    RevertPrepareBinaryForLLGA(g);
    GRAPH_DUMP("After RevertPrepareBinaryForLLGA. Before IpexQuantFusion", g);
    IpexQuantFusion(g);
    GRAPH_DUMP("After IpexQuantFusion. End of INT8 optimization pass", g);
  }
}

void setLlgaWeightCacheEnabled(bool enabled) {
  dnnl::graph::set_constant_tensor_cache(enabled);
}

bool getLlgaWeightCacheEnabled() {
  return dnnl::graph::get_constant_tensor_cache();
}

} // namespace onednn
} // namespace fuser

using namespace torch::jit;

Operation createLlgaKernel(const Node* node) {
  auto kernel = std::make_shared<fuser::onednn::LlgaKernel>(node);
  return [kernel](Stack* stack) {
    RECORD_FUNCTION(kernel->profileName(), c10::ArrayRef<c10::IValue>());

    kernel->run(*stack);
    return 0;
  };
}

torch::jit::RegisterOperators LLGAFusionGroupOp({
    torch::jit::Operator(
        Symbol::fromQualString(fuser::onednn::LlgaFusionGroupName()),
        createLlgaKernel,
        AliasAnalysisKind::PURE_FUNCTION),
});

} // namespace jit
} // namespace torch_ipex
