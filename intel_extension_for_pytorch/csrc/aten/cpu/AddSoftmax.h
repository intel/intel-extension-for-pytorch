#pragma once

#include <ATen/ATen.h>
#include <csrc/dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

// This operator assumes that the softmax is applied to the last
// dimension.
at::Tensor DivAddSoftmax(
    at::Tensor& a,
    const at::Tensor& b,
    const float& dim_per_head);

#if defined(DYN_DISP_BUILD)
namespace {
#endif

at::Tensor div_add_softmax_kernel_impl(
    at::Tensor& a,
    const at::Tensor& b,
    const float& dim_per_head);

#if defined(DYN_DISP_BUILD)
}
#endif

using div_add_softmax_kernel_fn =
    at::Tensor (*)(at::Tensor&, const at::Tensor&, const float&);
DECLARE_DISPATCH(div_add_softmax_kernel_fn, div_add_softmax_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
