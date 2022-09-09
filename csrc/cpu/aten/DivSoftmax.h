#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

// This operator assumes that the softmax is applied to the last
// dimension.
at::Tensor DivMaskedfillSoftmax(
    at::Tensor& a,
    const at::Tensor& b,
    const at::IntArrayRef& mask_shape,
    const float& fill,
    const float& dim_per_head);

namespace {

at::Tensor div_maskedfill_softmax_kernel_impl(
    at::Tensor& a,
    const at::Tensor& b,
    const at::IntArrayRef& mask_shape,
    const float& fill,
    const float& dim_per_head);

}

using div_maskedfill_softmax_kernel_fn = at::Tensor (*)(
    at::Tensor&,
    const at::Tensor&,
    const at::IntArrayRef&,
    const float&,
    const float&);
DECLARE_DISPATCH(
    div_maskedfill_softmax_kernel_fn,
    div_maskedfill_softmax_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
