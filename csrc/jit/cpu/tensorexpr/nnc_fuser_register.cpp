#include "nnc_fuser_register.h"

#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include "operator_schema.h"

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

void clearCustomOp2NncFuser() {
  auto& _g_custom_operator_set = torch::jit::tensorexpr::getCustomOperatorSet();
  _g_custom_operator_set = {};
}

void registerCustomOp2NncFuser() {
  auto& _g_custom_operator_set = torch::jit::tensorexpr::getCustomOperatorSet();
  _g_custom_operator_set.insert(
      {kMmDivSchema,       kConvNoneSchema,        kConvReluSchema,
       kConvAbsSchema,     kConvClampSchema,       kConvEluSchema,
       kConvExpSchema,     kConvGeluSchema,        kConvHardswishSchema,
       kConvLogSchema,     kConvMishSchema,        kConvSigmoidSchema,
       kConvPowSchema,     kConvRoundSchema,       kConvSqrtSchema,
       kConvSquareSchema,  kConvTanhSchema,        kConvLeakyReluSchema,
       kConvSiluSchema,    kConvHardsigmoidSchema, kLinearNoneSchema,
       kLinearAbsSchema,   kLinearExpSchema,       kLinearHardswishSchema,
       kLinearMishSchema,  kLinearSigmoidSchema,   kLinearReluSchema,
       kLinearSqrtSchema,  kLinearSquareSchema,    kLinearTanhSchema,
       kLinearSiluSchema,  kLinearLogSchema,       kLinearRoundSchema,
       kLinearClampSchema, kLinearEluSchema,       kLinearGeluSchema,
       kLinearPowSchema,   kLinearLeakyReluSchema, kLinearHardsigmoidSchema});
}

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
