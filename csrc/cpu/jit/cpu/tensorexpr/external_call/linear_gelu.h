#pragma once

#include <ideep.hpp>
#include <ideep/utils.hpp>

#include "linear_common.h"

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

static std::unordered_map<std::string, int64_t> approximate_str2int_map = {
    {"none", 0},
    {"tanh", 1}};

static std::unordered_map<int64_t, dnnl::algorithm> apprroximate_int2alg_map = {
    {0, dnnl::algorithm::eltwise_gelu_erf},
    {1, dnnl::algorithm::eltwise_gelu_tanh}};

template <>
struct LoweringFuncTrait<LinearFusedOp::kLinearGelu>
    : public LinearCommonOperations {
  DECLARE_LINEAR_FUNC_AND_RES(gelu)

  /**
   * @note This operator fuses linear and gelu.
   *
   * Its schema is  "ipex_prepack::linear_gelu_run(
   *  Tensor input,
   *  *,
   *  str approximate,
   *  __torch__.torch.classes.ipex_prepack.LinearOpContext W_prepack) ->
   * Tensor"
   *
   */

  static std::vector<pytnnc::BufHandle> get_input_buf(
      const std::vector<pytnnc::ArgValue>& inputs) {
    std::vector<pytnnc::BufHandle> res = {};
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inputs.size() == 3);
    // The order is:
    //     0: activator tensor
    //     1: approximate
    //     2: linear op context
    constexpr int input_idx = 0; // input tensor
    constexpr int ctx_idx = 2; // Linear context
    res.push_back(c10::get<pytnnc::BufHandle>(inputs[input_idx]));
    res.push_back(c10::get<pytnnc::BufHandle>(inputs[ctx_idx]));
    return res;
  }

  static std::vector<pytnnc::ExprHandle> get_extra_args(
      const std::vector<pytnnc::ArgValue>& inputs) {
    constexpr int approximate_idx = 1;
    std::vector<pytnnc::ExprHandle> extra_args;
    auto approximate = c10::get_if<std::string>(&inputs[approximate_idx]);
    if (approximate_str2int_map.find(*approximate) ==
        approximate_str2int_map.end()) {
      TORCH_CHECK(false, "linear_gelu only support tanh approximate now");
    }
    return {static_cast<int64_t>(approximate_str2int_map[*approximate])};
  }

  static ideep::attr_t get_attr(int64_t* extra_args) {
    constexpr int approximate_idx = 0;
    const int64_t approximate_int =
        static_cast<int64_t>(((int64_t*)extra_args)[approximate_idx]);
    dnnl::algorithm gelu_type = apprroximate_int2alg_map[approximate_int];
    return ideep::attr_t::fuse_gelu(1.0, 0.f, 0.f, gelu_type);
  }
};

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
