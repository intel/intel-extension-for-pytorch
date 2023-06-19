#pragma once

#include "conv_common.h"

#include <ATen/ATen.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

template <>
struct LoweringFuncTrait<ConvFusedOp::kConvBottleneckV2>
    : public ConvCommonOperations {
  DECLARE_CONV_FUNC_AND_RES(bottleneck_v2)

  static std::vector<pytnnc::BufHandle> get_input_buf(
      const std::vector<pytnnc::ArgValue>& inputs) {
    std::vector<pytnnc::BufHandle> res = {};
    auto buf_num = inputs.size();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY((buf_num == 5) || (buf_num == 4));
    // The order is:
    //     0: activator(in) tensor
    //     1: conv op context
    //     2: conv op context
    //     3: conv op context
    for (int i = 0; i < buf_num; i++) {
      res.push_back(c10::get<pytnnc::BufHandle>(inputs[i]));
    }
    return res;
  }
};

template <>
struct LoweringFuncTrait<ConvFusedOp::kConvBottleneckV1>
    : public LoweringFuncTrait<ConvFusedOp::kConvBottleneckV2> {
  DECLARE_CONV_FUNC_AND_RES(bottleneck_v1)

  static pytnnc::BufHandle get_result_buf(
      const char* res_var_name,
      const std::vector<pytnnc::ArgValue>& inputs,
      const std::vector<pytnnc::ExprHandle>& output_shape,
      const std::vector<pytnnc::ExprHandle>& output_strides,
      const c10::optional<pytnnc::ScalarType>& output_type) {
    // The order is:
    //     0: activator(in/out) tensor
    //     1: conv op context
    //     2: conv op context
    //     3: conv op context
    constexpr int res_idx = 0;
    return c10::get<pytnnc::BufHandle>(inputs[res_idx]);
  }
};

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
