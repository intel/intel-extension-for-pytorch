#pragma once

#include <ATen/Tensor.h>
#include <dyndisp/DispatchStub.h>
#include <torch/csrc/autograd/custom_function.h>
#include <vector>

#include <ideep.hpp>
#include "cpu/kernels/OpContext.h"

namespace torch_ipex {
namespace cpu {

void linear_kernel_output(
    const at::Tensor& self,
    const ideep::tensor& mkldnn_weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const ideep::attr_t& attr,
    const std::vector<ideep::tensor>& post_op_src = {});

at::Tensor linear_kernel(
    const at::Tensor& self,
    const ideep::tensor& mkldnn_weight,
    const at::Tensor& bias,
    const ideep::attr_t& attr,
    const std::vector<ideep::tensor>& post_op_src = {});

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_backward_kernel(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    std::array<bool, 3> output_mask,
    ideep::tensor packed_weight,
    const c10::optional<at::Tensor>& bias);

// IPEX customized linear OP with n-D packed weight
// Additional out_features, in_features is used to query expected weigth desc
// Since n-D packed weight have loss these info
class IPEXLinearOp : public torch::autograd::Function<IPEXLinearOp> {
 public:
  // forward function without autograd overhead, will go this way when only do
  // forward
  static at::Tensor _forward(
      const at::Tensor& input,
      const at::Tensor& weight,
      const c10::optional<at::Tensor>& bias,
      const int64_t eltwise,
      const at::Tensor& op_context,
      const c10::optional<int64_t> out_features);

  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const at::Tensor& weight,
      const c10::optional<at::Tensor>& bias,
      const int64_t eltwise,
      const at::Tensor& op_context,
      const c10::optional<int64_t> out_features);

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs);
};

at::Tensor ipex_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& op_context,
    const c10::optional<int64_t> out_features);

at::Tensor ipex_linear_eltwise(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const int64_t eltwise,
    const at::Tensor& op_context,
    const c10::optional<int64_t> out_features);

#ifdef USE_LIBXSMM
// WOQ linear ops
at::Tensor woq_linear_forward(
    const at::Tensor& input,
    const at::Tensor& op_context);

at::Tensor woq_linear_forward_v2(
    const at::Tensor& input,
    const at::Tensor& qweight,
    const c10::string_view& weight_dtype,
    const std::vector<int64_t>& weight_shape,
    const std::vector<at::Tensor>& weight_scales,
    const c10::optional<std::vector<at::Tensor>>& weight_zeros,
    const c10::optional<std::vector<at::Tensor>>& bias,
    const c10::optional<at::Tensor>& g_idx,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t act_quant_mode,
    const c10::optional<at::Tensor>& compensation);

at::Tensor woq_linear_gelu_forward(
    const at::Tensor& input,
    const at::Tensor& op_context);

at::Tensor woq_linear_new_gelu_forward(
    const at::Tensor& input,
    const at::Tensor& op_context);

at::Tensor woq_linear_relu_forward(
    const at::Tensor& input,
    const at::Tensor& op_context);

at::Tensor woq_linear_silu_forward(
    const at::Tensor& input,
    const at::Tensor& op_context);

at::Tensor woq_linear_add_forward(
    const at::Tensor& input,
    const at::Tensor& op_context,
    const std::vector<at::Tensor>& others);

at::Tensor woq_linear_add_add_forward(
    const at::Tensor& input,
    const at::Tensor& op_context,
    const std::vector<at::Tensor>& others);

at::Tensor woq_linear_mul_forward(
    const at::Tensor& input,
    const at::Tensor& op_context,
    const std::vector<at::Tensor>& others);

at::Tensor woq_linear_pack_weight(
    const at::Tensor& weight,
    int64_t weight_dtype,
    std::vector<int64_t>& weight_shape,
    int64_t group_size,
    int64_t lowp_mode);

at::Tensor woq_linear_compute_compensation(
    const at::Tensor& weight,
    int64_t weight_dtype,
    int64_t group_size,
    int64_t lowp_mode);

at::Tensor woq_linear_unpack_weight(
    const at::Tensor& weight,
    int64_t weight_dtype,
    int64_t lowp_mode);

at::Tensor woq_linear_kernel(
    const at::Tensor& self,
    const at::Tensor& weight,
    int64_t weight_dtype,
    const std::vector<at::Tensor>& scales_list,
    const std::vector<at::Tensor>& zps_list,
    const std::vector<at::Tensor>& bias_list,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t act_quant_mode,
    const c10::optional<at::Tensor>& compensation = c10::nullopt);

at::Tensor woq_linear_unary_kernel(
    const at::Tensor& self,
    const at::Tensor& weight,
    int64_t weight_dtype,
    const std::vector<at::Tensor>& scales_list,
    const std::vector<at::Tensor>& zps_list,
    const std::vector<at::Tensor>& bias_list,
    const c10::string_view& post_op,
    const torch::List<c10::optional<at::Scalar>>& scalars,
    const c10::optional<c10::string_view>& algorithm,
    int64_t group_size,
    int64_t lowp_mode,
    int64_t act_quant_mode,
    const c10::optional<at::Tensor>& compensation = c10::nullopt);

at::Tensor woq_linear_binary_kernel(
    const at::Tensor& self,
    const at::Tensor& weight,
    int64_t weight_dtype,
    const std::vector<at::Tensor>& scales_list,
    const std::vector<at::Tensor>& zps_list,
    const std::vector<at::Tensor>& bias_list,
    int64_t group_size,
    int64_t lowp_mode,
    const c10::string_view& post_op,
    const std::vector<at::Tensor>& others,
    int64_t act_quant_mode,
    const c10::optional<at::Tensor>& compensation = c10::nullopt);

namespace {
void woq_gemm_kernel_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& scales_float,
    const at::Tensor& zero_points_float,
    const at::Tensor& bias,
    int64_t lowp_mode,
    at::Tensor& output);

void woq_gemm_eltwise_kernel_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& scales_float,
    const at::Tensor& zero_points_float,
    const at::Tensor& bias,
    const c10::string_view& post_op,
    const torch::List<c10::optional<at::Scalar>>& scalars,
    const c10::optional<c10::string_view>& algorithm,
    int64_t lowp_mode,
    at::Tensor& output);

at::Tensor woq_linear_packB_impl(
    const at::Tensor& weight,
    const at::Tensor& scales,
    const at::Tensor& zero_points);

at::Tensor woq_linear_unpackB_impl(const at::Tensor& weight);

} // namespace

using woq_tpp_gemm_kernel_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const int,
    int64_t,
    int64_t,
    const std::vector<at::Tensor>&,
    int64_t,
    int64_t,
    int64_t,
    const c10::optional<at::Tensor>&);

using woq_gemm_kernel_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const int,
    int64_t,
    const std::vector<at::Tensor>&,
    int64_t,
    int64_t);

using woq_int8_gemm_kernel_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const int,
    int64_t,
    const std::vector<at::Tensor>&,
    int64_t,
    int64_t,
    int64_t,
    const c10::optional<at::Tensor>&);

using woq_tpp_gemm_packB_fn =
    at::Tensor (*)(const at::Tensor&, int, size_t, size_t, int64_t);

using woq_tpp_gemm_unpackB_fn = at::Tensor (*)(const at::Tensor&, int, int64_t);

using woq_dequant_int4_to_int8_packed_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    int,
    int64_t,
    at::Tensor&);

using dequant_nf4_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    int64_t,
    c10::ScalarType);

IPEX_DECLARE_DISPATCH(woq_tpp_gemm_kernel_fn, woq_tpp_gemm_kernel_stub);
IPEX_DECLARE_DISPATCH(woq_gemm_kernel_fn, woq_fp32_gemm_kernel_stub);
IPEX_DECLARE_DISPATCH(woq_gemm_kernel_fn, woq_fp16_gemm_kernel_stub);
IPEX_DECLARE_DISPATCH(woq_gemm_kernel_fn, woq_bf16_gemm_kernel_stub);
IPEX_DECLARE_DISPATCH(
    woq_int8_gemm_kernel_fn,
    woq_int8_gemm_pre_tensor_kernel_stub);
IPEX_DECLARE_DISPATCH(
    woq_int8_gemm_kernel_fn,
    woq_int8_gemm_pre_k_block_kernel_stub);
IPEX_DECLARE_DISPATCH(
    woq_int8_gemm_kernel_fn,
    woq_int8_gemm_pre_m_block_kernel_stub);
IPEX_DECLARE_DISPATCH(
    woq_int8_gemm_kernel_fn,
    woq_int8_gemm_pre_m_k_block_kernel_stub);
IPEX_DECLARE_DISPATCH(woq_tpp_gemm_packB_fn, woq_tpp_gemm_packB_stub);
IPEX_DECLARE_DISPATCH(woq_tpp_gemm_unpackB_fn, woq_tpp_gemm_unpackB_stub);
IPEX_DECLARE_DISPATCH(
    woq_dequant_int4_to_int8_packed_fn,
    woq_dequant_int4_to_int8_packed_stub);
IPEX_DECLARE_DISPATCH(dequant_nf4_fn, dequant_nf4_stub);

#endif

} // namespace cpu
} // namespace torch_ipex
