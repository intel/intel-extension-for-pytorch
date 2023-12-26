#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>
#include <cstdint>

namespace torch_ipex {

namespace cpu {

using forward_fn = void (*)(
    const at::Tensor& /* X */,
    const at::Tensor& /* gamma */,
    const at::Tensor& /* beta */,
    int64_t /* N */,
    int64_t /* C */,
    int64_t /* HxW */,
    int64_t /* group */,
    double /* eps */,
    at::Tensor& /* Y */,
    at::Tensor& /* mean */,
    at::Tensor& /* rstd */);

using backward_fn = void (*)(
    const at::Tensor& /* dY */,
    const at::Tensor& /* X */,
    const at::Tensor& /* mean */,
    const at::Tensor& /* rstd */,
    const at::Tensor& /* gamma */,
    int64_t /* N */,
    int64_t /* C */,
    int64_t /* HxW */,
    int64_t /* group */,
    at::Tensor& /* dX */,
    at::Tensor& /* dgamma */,
    at::Tensor& /* dbeta */);

IPEX_DECLARE_DISPATCH(forward_fn, GroupNormKernel);
IPEX_DECLARE_DISPATCH(backward_fn, GroupNormBackwardKernel);

} // namespace cpu
} // namespace torch_ipex