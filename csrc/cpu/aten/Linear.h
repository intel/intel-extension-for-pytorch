#pragma once

#include <ATen/Tensor.h>
#include <dyndisp/DispatchStub.h>
#include <torch/csrc/autograd/custom_function.h>
#include <vector>

#include <ideep.hpp>
#include "cpu/kernels/OpContext.h"

namespace torch_ipex {
namespace cpu {

at::Tensor woq_linear_forward(
    const at::Tensor& input,
    const at::Tensor& op_context);

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
at::Tensor woq_linear_pack_weight(
    const at::Tensor& weight,
    int64_t weight_dtype,
    std::vector<int64_t>& weight_shape,
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
    int64_t act_quant_mode);

at::Tensor woq_linear_eltwise_kernel(
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
    int64_t act_quant_mode);

at::Tensor woq_linear_add_kernel(
    const at::Tensor& self,
    const at::Tensor& weight,
    int64_t weight_dtype,
    const std::vector<at::Tensor>& scales_list,
    const std::vector<at::Tensor>& zps_list,
    const std::vector<at::Tensor>& bias_list,
    int64_t group_size,
    int64_t lowp_mode,
    const std::vector<at::Tensor>& others,
    int64_t act_quant_mode);

at::Tensor woq_linear_add_add_kernel(
    const at::Tensor& self,
    const at::Tensor& weight,
    int64_t weight_dtype,
    const std::vector<at::Tensor>& scales_list,
    const std::vector<at::Tensor>& zps_list,
    const std::vector<at::Tensor>& bias_list,
    int64_t group_size,
    int64_t lowp_mode,
    const std::vector<at::Tensor>& others,
    int64_t act_quant_mode);

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

using woq_gemm_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    int64_t,
    at::Tensor&);

using woq_gemm_eltwise_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const c10::string_view&,
    const torch::List<c10::optional<at::Scalar>>&,
    const c10::optional<c10::string_view>&,
    int64_t,
    at::Tensor&);

using woq_linear_packB_fn =
    at::Tensor (*)(const at::Tensor&, const at::Tensor&, const at::Tensor&);

using woq_linear_unpackB_fn = at::Tensor (*)(const at::Tensor&);

IPEX_DECLARE_DISPATCH(woq_gemm_kernel_fn, woq_gemm_kernel_stub);
IPEX_DECLARE_DISPATCH(woq_gemm_eltwise_kernel_fn, woq_gemm_eltwise_kernel_stub);
IPEX_DECLARE_DISPATCH(woq_linear_packB_fn, woq_linear_packB_stub);
IPEX_DECLARE_DISPATCH(woq_linear_unpackB_fn, woq_linear_unpackB_stub);

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
    int64_t);

using woq_tpp_gemm_packB_fn =
    at::Tensor (*)(const at::Tensor&, int, size_t, size_t, int64_t);

using woq_tpp_gemm_unpackB_fn = at::Tensor (*)(const at::Tensor&, int, int64_t);

IPEX_DECLARE_DISPATCH(woq_tpp_gemm_kernel_fn, woq_tpp_gemm_kernel_stub);
IPEX_DECLARE_DISPATCH(woq_tpp_gemm_packB_fn, woq_tpp_gemm_packB_stub);
IPEX_DECLARE_DISPATCH(woq_tpp_gemm_unpackB_fn, woq_tpp_gemm_unpackB_stub);

#define WOQ_FUSE_NONE 0
#define WOQ_FUSE_GELU 1
#define WOQ_FUSE_ADD 2
#define WOQ_FUSE_ADD_ADD 3
#define WOQ_FUSE_NEW_GELU 4

#endif

} // namespace cpu
} // namespace torch_ipex
