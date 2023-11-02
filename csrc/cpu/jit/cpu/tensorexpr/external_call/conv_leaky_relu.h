#pragma once

#include <ideep.hpp>
#include <ideep/utils.hpp>

#include "conv_common.h"

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

template <>
struct LoweringFuncTrait<ConvFusedOp::kConvLeakyRelu>
    : public ConvCommonOperations {
  DECLARE_CONV_FUNC_AND_RES(leaky_relu)

  /**
   * @note This operator fuses conv and leaky relu.
   *
   * Its schema is  "ipex_prepack::convolution_leaky_relu_run(
   *  Tensor input,
   *  *,
   *  Scalar alpha,
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
    //     1: alpha
    //     2: conv op context
    constexpr int input_idx = 0; // input tensor
    constexpr int ctx_idx = 2; // Conv context
    res.push_back(std::get<pytnnc::BufHandle>(inputs[input_idx]));
    res.push_back(std::get<pytnnc::BufHandle>(inputs[ctx_idx]));
    return res;
  }

  static std::vector<pytnnc::ExprHandle> get_extra_args(
      const std::vector<pytnnc::ArgValue>& inputs) {
    constexpr int alpha_idx = 1;
    std::vector<pytnnc::ExprHandle> extra_args;
    insert_scalar_arg(inputs[alpha_idx], extra_args);
    return extra_args;
  }

  static ideep::attr_t get_attr(int64_t* extra_args) {
    constexpr int alpha_idx = 0;
    const float alpha = static_cast<float>(((double*)extra_args)[alpha_idx]);
    return ideep::attr_t::fuse_relu(1.0, alpha);
  }
};

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
