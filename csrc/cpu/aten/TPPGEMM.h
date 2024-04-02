#pragma once
#ifdef USE_LIBXSMM
#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

at::Tensor tpp_linear_nobias_forward_cpu(
    const at::Tensor& t_in,
    const at::Tensor& t_wt,
    c10::optional<int64_t> out_features);

at::Tensor tpp_linear_bias_forward_cpu(
    const at::Tensor& t_in,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias,
    c10::optional<int64_t> out_features);

at::Tensor tpp_linear_gelu_forward_cpu(
    const at::Tensor& t_in,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias,
    c10::optional<int64_t> out_features);

at::Tensor tpp_fused_gate_up_proj_forward_cpu(
    const at::Tensor& t_in,
    const at::Tensor& t_wt_gate,
    const at::Tensor& t_bias_gate,
    const at::Tensor& t_wt_up,
    const at::Tensor& t_bias_up,
    c10::optional<int64_t> out_features);

at::Tensor tpp_linear_silu_forward_cpu(
    const at::Tensor& t_in,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias,
    c10::optional<int64_t> out_features);

at::Tensor tpp_linear_relu_forward_cpu(
    const at::Tensor& t_in,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias,
    c10::optional<int64_t> out_features);

at::Tensor tpp_linear_add_forward_cpu(
    const at::Tensor& t_in,
    const at::Tensor& t_in1,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias,
    double scale,
    c10::optional<int64_t> out_features);

at::Tensor tpp_linear_mul_forward_cpu(
    const at::Tensor& t_in,
    const at::Tensor& t_in1,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias,
    c10::optional<int64_t> out_features);

at::Tensor tpp_linear_add_add_forward_cpu(
    const at::Tensor& t_in,
    const at::Tensor& t_in1,
    const at::Tensor& t_in2,
    const at::Tensor& t_wt,
    const at::Tensor& t_bias,
    double scale,
    c10::optional<int64_t> out_features);

using tpp_linear_nobias_impl_fn =
    at::Tensor (*)(const at::Tensor&, const at::Tensor&);

using tpp_linear_bias_kernel_impl_fn =
    at::Tensor (*)(const at::Tensor&, const at::Tensor&, const at::Tensor&);

using tpp_linear_gelu_kernel_impl_fn =
    at::Tensor (*)(const at::Tensor&, const at::Tensor&, const at::Tensor&);

using tpp_fused_gate_up_proj_kernel_impl_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&);

using tpp_linear_silu_kernel_impl_fn =
    at::Tensor (*)(const at::Tensor&, const at::Tensor&, const at::Tensor&);

using tpp_linear_relu_kernel_impl_fn =
    at::Tensor (*)(const at::Tensor&, const at::Tensor&, const at::Tensor&);

using tpp_linear_add_kernel_impl_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double);

using tpp_linear_mul_kernel_impl_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&);

using tpp_linear_add_add_kernel_impl_fn = at::Tensor (*)(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double);

IPEX_DECLARE_DISPATCH(tpp_linear_nobias_impl_fn, tpp_linear_nobias_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_linear_bias_kernel_impl_fn,
    tpp_linear_bias_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_linear_gelu_kernel_impl_fn,
    tpp_linear_gelu_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_fused_gate_up_proj_kernel_impl_fn,
    tpp_fused_gate_up_proj_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_linear_silu_kernel_impl_fn,
    tpp_linear_silu_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_linear_relu_kernel_impl_fn,
    tpp_linear_relu_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_linear_add_kernel_impl_fn,
    tpp_linear_add_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_linear_mul_kernel_impl_fn,
    tpp_linear_mul_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_linear_add_add_kernel_impl_fn,
    tpp_linear_add_add_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
#endif