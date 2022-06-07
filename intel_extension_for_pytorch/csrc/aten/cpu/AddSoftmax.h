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

namespace {

at::Tensor div_add_softmax_kernel_impl(
    at::Tensor& a,
    const at::Tensor& b,
    const float& dim_per_head);
}

using div_add_softmax_kernel_fn =
    at::Tensor (*)(at::Tensor&, const at::Tensor&, const float&);
using add_softmax_inplace_kernel_fn =
    at::Tensor& (*)(at::Tensor&, const at::Tensor&);
DECLARE_DISPATCH(div_add_softmax_kernel_fn, div_add_softmax_kernel_stub);
DECLARE_DISPATCH(
    add_softmax_inplace_kernel_fn,
    add_softmax_inplace_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
