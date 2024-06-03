#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

at::Tensor dil_RMSNorm(
    const at::Tensor& input,
    const at::Tensor& b,
    double eps);

namespace {

at::Tensor rmsnorm_kernel_impl(
    const at::Tensor& input,
    const at::Tensor& b,
    float eps);

at::Tensor add_rmsnorm_kernel_impl(
    const at::Tensor& input,
    at::Tensor& input1,
    const at::Tensor& b,
    float eps,
    bool add_back); // if true, store sum in input1
} // namespace

using rms_norm_kernel_fn =
    at::Tensor (*)(const at::Tensor&, const at::Tensor&, float);
using add_rms_norm_kernel_fn = at::Tensor (*)(
    const at::Tensor&,
    at::Tensor&,
    const at::Tensor&,
    float,
    bool);

IPEX_DECLARE_DISPATCH(rms_norm_kernel_fn, rmsnorm_kernel_stub);
IPEX_DECLARE_DISPATCH(add_rms_norm_kernel_fn, add_rmsnorm_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
