#pragma once

#include <ideep.hpp>
#include <ideep/utils.hpp>

#include "linear_common.h"

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

template <>
struct LoweringFuncTrait<LinearFusedOp::kLinearNone>
    : public LinearCommonOperations {
  DECLARE_LINEAR_FUNC_AND_RES(none)
};

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
