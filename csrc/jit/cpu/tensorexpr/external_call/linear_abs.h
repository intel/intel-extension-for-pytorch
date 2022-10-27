#pragma once

#include <ideep.hpp>
#include <ideep/utils.hpp>

#include "linear_common.h"

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

template <>
struct LoweringFuncTrait<LinearFusedOp::kLinearAbs>
    : public LinearCommonOperations {
  DECLARE_LINEAR_FUNC_AND_RES(abs)

  static ideep::attr_t get_attr(int64_t* buf_data) {
    return ideep::attr_t::fuse_abs();
  }
};

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
