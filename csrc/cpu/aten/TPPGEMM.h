#pragma once
#ifdef USE_LIBXSMM
#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

namespace {
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

} // namespace

using tpp_linear_nobias_impl_fn =
    at::Tensor (*)(const at::Tensor&, const at::Tensor&);

using tpp_linear_bias_kernel_impl_fn =
    at::Tensor (*)(const at::Tensor&, const at::Tensor&, const at::Tensor&);

using tpp_linear_gelu_kernel_impl_fn =
    at::Tensor (*)(const at::Tensor&, const at::Tensor&, const at::Tensor&);

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

DECLARE_DISPATCH(tpp_linear_nobias_impl_fn, tpp_linear_nobias_kernel_stub);
DECLARE_DISPATCH(tpp_linear_bias_kernel_impl_fn, tpp_linear_bias_kernel_stub);
DECLARE_DISPATCH(tpp_linear_gelu_kernel_impl_fn, tpp_linear_gelu_kernel_stub);
DECLARE_DISPATCH(tpp_linear_silu_kernel_impl_fn, tpp_linear_silu_kernel_stub);
DECLARE_DISPATCH(tpp_linear_relu_kernel_impl_fn, tpp_linear_relu_kernel_stub);
DECLARE_DISPATCH(tpp_linear_add_kernel_impl_fn, tpp_linear_add_kernel_stub);
DECLARE_DISPATCH(tpp_linear_mul_kernel_impl_fn, tpp_linear_mul_kernel_stub);
DECLARE_DISPATCH(
    tpp_linear_add_add_kernel_impl_fn,
    tpp_linear_add_add_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
#endif