#pragma once

#include <ideep.hpp>
#include <ideep/utils.hpp>

#include "linear_common.h"

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

template <>
struct LoweringFuncTrait<LinearFusedOp::kLinearExp>
    : public LinearCommonOperations {
  DECLARE_LINEAR_FUNC_AND_RES(exp)

  static ideep::attr_t get_attr(int64_t* buf_data) {
    return ideep::attr_t::fuse_exp();
  }
};

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
