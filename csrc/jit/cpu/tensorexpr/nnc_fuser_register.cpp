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

void registerCustomOp2NncFuser() {
  auto& _g_custom_operator_set = torch::jit::tensorexpr::getCustomOperatorSet();
  _g_custom_operator_set.insert(
      {kMmDivSchema,         kConvNoneSchema,   kConvReluSchema,
       kConvAddReluSchema,   kConvAbsSchema,    kConvClampSchema,
       kConvEluSchema,       kConvExpSchema,    kConvGeluSchema,
       kConvHardswishSchema, kConvLogSchema,    kConvMishSchema,
       kConvSigmoidSchema,   kConvPowSchema,    kConvRoundSchema,
       kConvSqrtSchema,      kConvSquareSchema, kConvTanhSchema,
       kConvLeakyReluSchema, kConvSiluSchema});
}

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
