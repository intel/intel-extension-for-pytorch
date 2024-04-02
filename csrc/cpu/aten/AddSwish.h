#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

// Currently we only support 1D tensor of bias(operand of add).
at::Tensor AddSwish(
    at::Tensor& x,
    at::Tensor& mm_output,
    const at::Tensor& weight,
    const at::Tensor& bias);

namespace {

at::Tensor add_swish_kernel_impl(
    at::Tensor& x,
    at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c);
}

using add_swish_kernel_fn = at::Tensor (*)(
    at::Tensor&,
    at::Tensor&,
    const at::Tensor&,
    const at::Tensor&);
IPEX_DECLARE_DISPATCH(add_swish_kernel_fn, add_swish_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
