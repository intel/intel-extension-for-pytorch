#include "jit/codegen/onednn/graph_fuser.h"
#include "jit/codegen/onednn/interface.h"
#include "jit/codegen/onednn/kernel.h"
#include "jit/codegen/onednn/layout_propagation.h"
#include "jit/codegen/onednn/prepare_binary.h"
#include "jit/codegen/onednn/defer_size_check.h"
#include "jit/codegen/onednn/prepare_dequant.h"
#include "jit/codegen/onednn/dtype_propagation.h"
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

void fuseGraph(std::shared_ptr<Graph>& g) {
  GRAPH_DUMP("Before mutation removal. Beginning of LLGA optimization pass", g);
  RemoveTensorMutation(g);
  RemoveListMutation(g);
  GRAPH_DUMP("After mutation removal. Before DecomposeOps", g);
  DecomposeOps(g);
  GRAPH_DUMP("After DecomposeOps. Before PrepareBinaryForLLGA", g);
  PrepareBinaryForLLGA(g);
  GRAPH_DUMP("After PrepareBinaryForLLGA. Before EliminateCommonSubexpression", g);
  EliminateCommonSubexpression(g);
  GRAPH_DUMP("After EliminateCommonSubexpression. Before PrepareDequantForLLGA", g);
  // PrepareDequantForLLGA must be placed after EliminateCommonSubexpression
  PrepareDequantForLLGA(g);
  GRAPH_DUMP("After PrepareDequantForLLGA. Before DeferSizeCheck", g);
  DeferSizeCheck(g);  
  GRAPH_DUMP("After DeferSizeCheck. Before CreateLlgaSubgraphs", g);
  // CreateLlgaSubgraphs must be placed after all the preparation passes above
  CreateLlgaSubgraphs(g);
  GRAPH_DUMP("After CreateLlgaSubgraphs. Before PropagateLayout", g);
  // PropagateLayout and PropagateDtype must be placed after CreateLlgaSubgraphs
  PropagateLayout(g);
  GRAPH_DUMP("After PropagateLayout. Before PropagateDtype", g);
  PropagateDtype(g);
  GRAPH_DUMP("After PropagateDtype. End of LLGA optimization pass", g);
}

} // namespace onednn
} // namespace fuser

Operation createLlgaKernel(const Node* node) {
  auto kernel = std::make_shared<fuser::onednn::LlgaKernel>(node);
  return [kernel](Stack* stack) {
    RECORD_FUNCTION(kernel->debugName(), std::vector<c10::IValue>());
    kernel->run(*stack);
    return 0;
  };
}

RegisterOperators LLGAFusionGroupOp({
    torch::jit::Operator(
        Symbol::fromQualString("ipex::LlgaFusionGroup"),
        createLlgaKernel,
        AliasAnalysisKind::PURE_FUNCTION),
});

} // namespace jit
} // namespace torch
