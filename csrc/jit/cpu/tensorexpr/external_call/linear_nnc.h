#pragma once

#include "csrc/jit/cpu/kernels/LinearPacked.h"
#include "csrc/jit/cpu/kernels/OpContext.h"
#include "csrc/jit/cpu/tensorexpr/utils.h"

#include <ATen/ATen.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>

#include <vector>

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

namespace pytnnc = torch::jit::tensorexpr;

template <typename LoweringFunc>
pytnnc::Tensor computeLinear(
    const std::vector<pytnnc::ArgValue>& inputs,
    const std::vector<pytnnc::ExprHandle>& output_shape,
    const std::vector<pytnnc::ExprHandle>& output_strides,
    const c10::optional<pytnnc::ScalarType>& output_type,
    at::Device device) {
  pytnnc::BufHandle result_buf = LoweringFunc::get_result_buf(
      LoweringFunc::get_res_var(),
      inputs,
      output_shape,
      output_strides,
      output_type);
  pytnnc::StmtPtr s = pytnnc::ExternalCall::make(
      result_buf,
      LoweringFunc::get_external_func(),
      LoweringFunc::get_input_buf(inputs),
      LoweringFunc::get_extra_args(inputs));
  return pytnnc::Tensor(result_buf.node(), s);
}

template <typename LoweringFunc>
void nncLinear(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  // TODO: Need to take multi output buffer into account.
  constexpr int output_buf_idx = 0;
  constexpr int input_buf_idx = 1;

  int64_t buf_dims_idx = 0;
  int64_t buf_strides_idx = 0;
  std::vector<int64_t> buf_dims_vec;
  std::vector<int64_t> buf_strides_vec;
  // Output buffer shape and strides
  for (const auto dim : c10::irange(buf_ranks[output_buf_idx])) {
    (void)dim;
    buf_dims_vec.push_back(buf_dims[buf_dims_idx++]);
    buf_strides_vec.push_back(buf_strides[buf_strides_idx++]);
  }

  auto op_context = LoweringFunc::get_linear_op_context(buf_data);
  std::vector<at::Tensor> tensors = constructTensors(
      bufs_num - 1, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);
  at::Tensor& output = tensors[output_buf_idx];
  const at::Tensor& activation = tensors[input_buf_idx];
  torch_ipex::cpu::detail::linear::run_core(
      op_context->get_context(),
      activation,
      output,
      LoweringFunc::get_attr(extra_args));
}

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
