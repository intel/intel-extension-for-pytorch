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

at::Tensor woq_linear_pack_weight(
    const at::Tensor& weight,
    const at::Tensor& zero_points,
    const at::Tensor& scale);

at::Tensor woq_linear_unpack_weight(const at::Tensor& weight);

void woq_linear_kernel_output(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& zero_points_float,
    const at::Tensor& scales_float,
    const at::Tensor& bias,
    at::Tensor& output);

at::Tensor woq_linear_kernel(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& zero_points_float,
    const at::Tensor& scales_float,
    const at::Tensor& bias);

namespace {
void woq_gemm_kernel_impl(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& zero_points_float,
    const at::Tensor& scales_float,
    const at::Tensor& bias,
    at::Tensor& output);

at::Tensor woq_linear_packB_impl(
    const at::Tensor& weight,
    const at::Tensor& zero_points,
    const at::Tensor& scales);

at::Tensor woq_linear_unpackB_impl(const at::Tensor& weight);

} // namespace

using woq_gemm_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&);

using woq_linear_packB_fn =
    at::Tensor (*)(const at::Tensor&, const at::Tensor&, const at::Tensor&);

using woq_linear_unpackB_fn = at::Tensor (*)(const at::Tensor&);

DECLARE_DISPATCH(woq_gemm_kernel_fn, woq_gemm_kernel_stub);
DECLARE_DISPATCH(woq_linear_packB_fn, woq_linear_packB_stub);
DECLARE_DISPATCH(woq_linear_unpackB_fn, woq_linear_unpackB_stub);
} // namespace cpu
} // namespace torch_ipex
