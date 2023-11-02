#pragma once

#include <ideep.hpp>
#include <ideep/utils.hpp>

#include "linear_common.h"

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

template <>
struct LoweringFuncTrait<LinearFusedOp::kLinearElu>
    : public LinearCommonOperations {
  DECLARE_LINEAR_FUNC_AND_RES(elu)

  /**
   * @note This operator fuses linear and elu.
   *
   * Its schema is  "ipex_prepack::linear_elu_run(
   *  Tensor input,
   *  *,
   *  Scalar alpha,
   *  Scalar scale,
   *  Scalar input_scale,
   *  __torch__.torch.classes.ipex_prepack.LinearOpContext W_prepack) ->
   * Tensor"
   *
   */

  static std::vector<pytnnc::BufHandle> get_input_buf(
      const std::vector<pytnnc::ArgValue>& inputs) {
    std::vector<pytnnc::BufHandle> res = {};
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inputs.size() == 5);
    // The order is:
    //     0: activator tensor
    //     1: alpha
    //     2: scale
    //     3: input_scale
    //     4: linear op context
    constexpr int input_idx = 0; // input tensor
    constexpr int ctx_idx = 4; // Linear context
    res.push_back(std::get<pytnnc::BufHandle>(inputs[input_idx]));
    res.push_back(std::get<pytnnc::BufHandle>(inputs[ctx_idx]));
    return res;
  }

  static std::vector<pytnnc::ExprHandle> get_extra_args(
      const std::vector<pytnnc::ArgValue>& inputs) {
    constexpr int alpha_idx = 1;
    constexpr int scale_idx = 2;
    constexpr int input_scale_idx = 3;
    std::vector<pytnnc::ExprHandle> extra_args;
    insert_scalar_arg(inputs[alpha_idx], extra_args);
    insert_scalar_arg(inputs[scale_idx], extra_args);
    insert_scalar_arg(inputs[input_scale_idx], extra_args);
    return extra_args;
  }

  static ideep::attr_t get_attr(int64_t* extra_args) {
    constexpr int alpha_idx = 0;
    constexpr int scale_idx = 1;
    constexpr int input_scale_idx = 2;
    const float alpha = static_cast<float>(((double*)extra_args)[alpha_idx]);
    const float scale = static_cast<float>(((double*)extra_args)[scale_idx]);
    const float input_scale =
        static_cast<float>(((double*)extra_args)[input_scale_idx]);
    return ideep::attr_t::fuse_elu(scale, alpha, input_scale);
  }
};

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
