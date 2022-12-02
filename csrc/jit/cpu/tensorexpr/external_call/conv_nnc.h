#pragma once

#include "csrc/jit/cpu/kernels/ConvPacked.h"
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
pytnnc::Tensor computeConv(
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
void nncConv(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  // TODO: Need to profile the tensor construction overhead. If the overhead is
  // not negligible, the conv should operate on raw buffer directly.
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

  auto op_context = LoweringFunc::get_conv_op_context(buf_data);
  if (deduce_memory_format(buf_strides_vec, buf_dims_vec) ==
      c10::MemoryFormat::ChannelsLast) {
    torch_ipex::cpu::detail::convolution::run_core_nhwc(
        op_context->get_context(),
        buf_data[input_buf_idx], // input buffer
        buf_data[output_buf_idx]); // output buffer
  } else {
    std::vector<at::Tensor> tensors = constructTensors(
        bufs_num - 1, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);
    at::Tensor& output = tensors[output_buf_idx];
    const at::Tensor& activation = tensors[input_buf_idx];
    torch_ipex::cpu::detail::convolution::run_core(
        op_context->get_context(), activation, output);
  }
}

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
