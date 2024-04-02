#pragma once

#include <ideep.hpp>
#include <ideep/utils.hpp>

#include "conv_common.h"

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

template <>
struct LoweringFuncTrait<ConvFusedOp::kConvClamp>
    : public ConvCommonOperations {
  DECLARE_CONV_FUNC_AND_RES(clamp)

  /**
   * @note This operator fuses conv and clamp.
   *
   * Its schema is  "ipex_prepack::convolution_hardtanh_run(
   *  Tensor input,
   *  *,
   *  Scalar lower_bound,
   *  Scalar upper_bound,
   *  __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) ->
   * Tensor"
   *
   */

  static std::vector<pytnnc::BufHandle> get_input_buf(
      const std::vector<pytnnc::ArgValue>& inputs) {
    std::vector<pytnnc::BufHandle> res = {};
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inputs.size() == 4);
    // The order is:
    //     0: activator tensor
    //     1: lower_bound
    //     2: upper_bound
    //     3: conv op context
    constexpr int input_idx = 0; // input tensor
    constexpr int ctx_idx = 3; // Conv context
    res.push_back(std::get<pytnnc::BufHandle>(inputs[input_idx]));
    res.push_back(std::get<pytnnc::BufHandle>(inputs[ctx_idx]));
    return res;
  }

  static std::vector<pytnnc::ExprHandle> get_extra_args(
      const std::vector<pytnnc::ArgValue>& inputs) {
    constexpr int lower_bound_idx = 1;
    constexpr int upper_bound_idx = 2;
    std::vector<pytnnc::ExprHandle> extra_args;
    insert_scalar_arg(inputs[lower_bound_idx], extra_args);
    insert_scalar_arg(inputs[upper_bound_idx], extra_args);
    return extra_args;
  }

  static ideep::attr_t get_attr(int64_t* extra_args) {
    constexpr int lower_bound_idx = 0;
    constexpr int upper_bound_idx = 1;
    const float lower_bound =
        static_cast<float>(((double*)extra_args)[lower_bound_idx]);
    const float upper_bound =
        static_cast<float>(((double*)extra_args)[upper_bound_idx]);
    return ideep::attr_t::fuse_clamp(lower_bound, upper_bound);
  }
};

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
