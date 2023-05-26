#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

namespace {
at::Tensor fc_in_kernel_impl(
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias);

at::Tensor fc_plain_kernel_impl(
    at::Tensor& t_in,
    at::Tensor& t_wt,
    at::Tensor& t_bias);


at::Tensor fc_out_kernel_impl(
    at::Tensor& t_in,
    at::Tensor& t_in1,
    at::Tensor& t_in2,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    double scale);

at::Tensor qkv_kernel_impl(at::Tensor& t_in, at::Tensor& t_wt);

} // namespace

using fc_in_kernel_impl_fn =
    at::Tensor (*)(at::Tensor&, at::Tensor&, at::Tensor&);

using fc_plain_kernel_impl_fn =
    at::Tensor (*)(at::Tensor&, at::Tensor&, at::Tensor&);

using fc_out_kernel_impl_fn = at::Tensor (*)(
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    double);

using qkv_kernel_impl_fn = at::Tensor (*)(at::Tensor&, at::Tensor&);

DECLARE_DISPATCH(fc_plain_kernel_impl_fn, fc_plain_kernel_stub);
DECLARE_DISPATCH(fc_in_kernel_impl_fn, fc_in_kernel_stub);
DECLARE_DISPATCH(fc_out_kernel_impl_fn, fc_out_kernel_stub);
DECLARE_DISPATCH(qkv_kernel_impl_fn, qkv_kernel_stub);

} // namespace cpu
} // namespace torch_ipex