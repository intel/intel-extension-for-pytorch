#pragma once

#include <ideep.hpp>
#include <ideep/utils.hpp>

#include "conv_common.h"

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

template <>
struct LoweringFuncTrait<ConvFusedOp::kConvSigmoid>
    : public ConvCommonOperations {
  DECLARE_CONV_FUNC_AND_RES(sigmoid)

  static ideep::attr_t get_attr(int64_t* buf_data) {
    return ideep::attr_t::fuse_sigmoid();
  }
};

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
