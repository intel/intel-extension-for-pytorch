#pragma once

#include <ATen/ATen.h>
#include <csrc/dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

// Currently we only support 1D tensor of bias(operand of add).
at::Tensor AddSwish(
    at::Tensor& x,
    at::Tensor& mm_output,
    const at::Tensor& weight,
    const at::Tensor& bias);

#if defined(DYN_DISP_BUILD)
namespace {
#endif

at::Tensor add_swish_kernel_impl(
    at::Tensor& x,
    at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c);

#if defined(DYN_DISP_BUILD)
}
#endif

using add_swish_kernel_fn = at::Tensor (*)(
    at::Tensor&,
    at::Tensor&,
    const at::Tensor&,
    const at::Tensor&);
DECLARE_DISPATCH(add_swish_kernel_fn, add_swish_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
