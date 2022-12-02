#pragma once

#include <ideep.hpp>
#include <ideep/utils.hpp>

#include "conv_common.h"

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

template <>
struct LoweringFuncTrait<ConvFusedOp::kConvPow> : public ConvCommonOperations {
  DECLARE_CONV_FUNC_AND_RES(pow)

  /**
   * @note This operator fuses conv and pow.
   *
   * Its schema is  "ipex_prepack::convolution_pow_run(
   *  Tensor input,
   *  *,
   *  Scalar exponent,
   *  __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) ->
   * Tensor"
   *
   */

  static std::vector<pytnnc::BufHandle> get_input_buf(
      const std::vector<pytnnc::ArgValue>& inputs) {
    std::vector<pytnnc::BufHandle> res = {};
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inputs.size() == 3);
    // The order is:
    //     0: activator tensor
    //     1: exponent
    //     2: conv op context
    constexpr int input_idx = 0; // input tensor
    constexpr int ctx_idx = 2; // Conv context
    res.push_back(c10::get<pytnnc::BufHandle>(inputs[input_idx]));
    res.push_back(c10::get<pytnnc::BufHandle>(inputs[ctx_idx]));
    return res;
  }

  static std::vector<pytnnc::ExprHandle> get_extra_args(
      const std::vector<pytnnc::ArgValue>& inputs) {
    constexpr int exponent_idx = 1;
    std::vector<pytnnc::ExprHandle> extra_args;
    insert_scalar_arg(inputs[exponent_idx], extra_args);
    return extra_args;
  }

  static ideep::attr_t get_attr(int64_t* extra_args) {
    constexpr int exponent_idx = 0;
    const float exponent =
        static_cast<float>(((int64_t*)extra_args)[exponent_idx]);
    return ideep::attr_t::fuse_pow(1.0, 1.0, exponent);
  }
};

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
