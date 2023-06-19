#pragma once

#include "csrc/cpu/jit/cpu/kernels/ConvPacked.h"
#include "csrc/cpu/jit/cpu/kernels/OpContext.h"
#include "csrc/cpu/jit/cpu/tensorexpr/utils.h"

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
  // Output buffer shape and strides
  std::vector<int64_t> output_buf_dims_vec;
  std::vector<int64_t> output_buf_strides_vec;
  for (const auto dim : c10::irange(buf_ranks[output_buf_idx])) {
    (void)dim;
    output_buf_dims_vec.push_back(buf_dims[buf_dims_idx++]);
    output_buf_strides_vec.push_back(buf_strides[buf_strides_idx++]);
  }
  // Input buffer shape and strides
  std::vector<int64_t> input_buf_dims_vec;
  std::vector<int64_t> input_buf_strides_vec;
  for (const auto dim : c10::irange(buf_ranks[input_buf_idx])) {
    (void)dim;
    input_buf_dims_vec.push_back(buf_dims[buf_dims_idx++]);
    input_buf_strides_vec.push_back(buf_strides[buf_strides_idx++]);
  }
  auto op_context = LoweringFunc::get_conv_op_context(buf_data);

  auto input_mem_format =
      deduce_memory_format(input_buf_strides_vec, input_buf_dims_vec);
  auto output_mem_format =
      deduce_memory_format(output_buf_strides_vec, output_buf_dims_vec);
  c10::ScalarType output_dtype =
      static_cast<c10::ScalarType>(buf_dtypes[output_buf_idx]);
  ideep::memory::data_type dst_dtype =
      op_context->get_context().conv_params_.pd.dst_desc().get_data_type();

  bool use_fast_path = false;
  if (input_buf_dims_vec ==
          op_context->get_context().conv_params_.pd.src_desc().get_dims() &&
      omp_get_max_threads() ==
          op_context->get_context().conv_params_.pd_use_threads &&
      ((output_dtype == at::ScalarType::BFloat16 &&
        dst_dtype == dnnl_data_type_t::dnnl_bf16) ||
       (output_dtype == at::ScalarType::Float &&
        dst_dtype == dnnl_data_type_t::dnnl_f32))) {
    use_fast_path = true;
  }

  if (input_mem_format == c10::MemoryFormat::ChannelsLast &&
      output_mem_format == c10::MemoryFormat::ChannelsLast && use_fast_path) {
    torch_ipex::cpu::detail::convolution::run_core_fast_path_nhwc(
        op_context->get_context(),
        buf_data[input_buf_idx], // input buffer
        buf_data[output_buf_idx]); // output buffer
  } else {
    std::vector<at::Tensor> tensors = constructTensors(
        bufs_num - 1, buf_data, buf_ranks, buf_dims, buf_strides, buf_dtypes);
    auto suggested_mem_format =
        op_context->get_context().weight_is_channels_last_
        ? ((buf_ranks[input_buf_idx] == 4) ? c10::MemoryFormat::ChannelsLast
                                           : c10::MemoryFormat::ChannelsLast3d)
        : c10::MemoryFormat::Contiguous;
    at::Tensor activation =
        at::native::contiguous(tensors[input_buf_idx], suggested_mem_format);
    at::Tensor output =
        at::native::contiguous(tensors[output_buf_idx], suggested_mem_format);
    if (use_fast_path) {
      torch_ipex::cpu::detail::convolution::run_core_fast_path(
          op_context->get_context(), activation, output);
    } else {
      torch_ipex::cpu::detail::convolution::run_core_fallback(
          op_context->get_context(),
          activation,
          output,
          LoweringFunc::get_attr(extra_args));
    }
    if (output.data_ptr() != tensors[output_buf_idx].data_ptr()) {
      tensors[output_buf_idx].copy_(output);
    }
  }
}

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex