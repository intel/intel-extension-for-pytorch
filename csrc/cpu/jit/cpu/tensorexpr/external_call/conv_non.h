#pragma once

#include <ideep.hpp>
#include <ideep/utils.hpp>

#include "conv_common.h"

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

template <>
struct LoweringFuncTrait<ConvFusedOp::kConvNone> : public ConvCommonOperations {
  DECLARE_CONV_FUNC_AND_RES(none)
};

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
