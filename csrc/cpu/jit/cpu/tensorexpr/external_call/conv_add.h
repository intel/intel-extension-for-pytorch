#pragma once

#include "conv_common.h"

#include <ideep.hpp>
#include <ideep/utils.hpp>
#include "csrc/cpu/jit/cpu/kernels/OpContext.h"

#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

template <>
struct LoweringFuncTrait<ConvFusedOp::kConvAdd> : public ConvCommonOperations {
  DECLARE_CONV_FUNC_AND_RES(add)

  /**
   * @note This fused conv operator is inplaced operator. It fuses conv and
   * add.
   *
   * Its schema is  "ipex_prepack::convolution_add_run(
   *  Tensor input,
   *  Tensor(a!) accumu,
   *  *,
   *  Scalar? alpha,
   *  __torch__.torch.classes.ipex_prepack.ConvolutionOpContext W_prepack) ->
   * Tensor"
   *
   */
  static std::vector<pytnnc::BufHandle> get_input_buf(
      const std::vector<pytnnc::ArgValue>& inputs) {
    std::vector<pytnnc::BufHandle> res = {};
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inputs.size() == 4);
    // The order is:
    //     0: activator(in) tensor
    //     1: accum(in/out) tensor
    //     2: alpha
    //     3: conv op context
    constexpr int act_idx = 0; // Activation tensor
    constexpr int accumu_idx = 1; // Accumulation tensor
    constexpr int ctx_idx = 3; // Conv context
    res.push_back(std::get<pytnnc::BufHandle>(inputs[act_idx]));
    res.push_back(std::get<pytnnc::BufHandle>(inputs[accumu_idx]));
    res.push_back(std::get<pytnnc::BufHandle>(inputs[ctx_idx]));
    return res;
  }

  static std::vector<pytnnc::ExprHandle> get_extra_args(
      const std::vector<pytnnc::ArgValue>& inputs) {
    constexpr int alpha_idx = 2;
    std::vector<pytnnc::ExprHandle> extra_args;
    insert_scalar_arg(inputs[alpha_idx], extra_args);
    return extra_args;
  }

  static pytnnc::BufHandle get_result_buf(
      const char* res_var_name,
      const std::vector<pytnnc::ArgValue>& inputs,
      const std::vector<pytnnc::ExprHandle>& output_shape,
      const std::vector<pytnnc::ExprHandle>& output_strides,
      const c10::optional<pytnnc::ScalarType>& output_type) {
    // The order is:
    //     0: activator(in) tensor
    //     1: accum(in/out) tensor
    //     2: alpha
    //     3: conv op context
    constexpr int res_idx = 1;
    return std::get<pytnnc::BufHandle>(inputs[res_idx]);
  }

  static ideep::attr_t get_attr(int64_t* extra_args) {
    constexpr int alpha_idx = 0;
    const float alpha = static_cast<float>(((double*)extra_args)[alpha_idx]);
    return ideep::attr_t::fuse_sum(alpha);
  }

  static torch_ipex::cpu::ConvolutionOpContext* get_conv_op_context(
      void** buf_data) {
    // The order is:
    //     0: output tensor
    //     1: activator tensor
    //     2: accum tensor
    //     3: conv op context
    constexpr int ctx_idx = 3;
    return reinterpret_cast<torch_ipex::cpu::ConvolutionOpContext*>(
        buf_data[ctx_idx]);
  }
};

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
