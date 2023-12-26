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
}

using rms_norm_kernel_fn =
    at::Tensor (*)(const at::Tensor&, const at::Tensor&, float);

IPEX_DECLARE_DISPATCH(rms_norm_kernel_fn, rmsnorm_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
