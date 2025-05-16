#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

namespace {

void punica_bgmv_shrink(
    at::Tensor& out,
    at::Tensor& input,
    at::Tensor& weights,
    at::Tensor& indicies,
    const double scale);

void punica_bgmv_expand(
    at::Tensor& out,
    at::Tensor& input,
    at::Tensor& weights,
    at::Tensor& indicies,
    bool add_inputs);

void punica_bgmv_expand_slice(
    at::Tensor& out,
    at::Tensor& input,
    at::Tensor& weights,
    at::Tensor& indicies,
    int64_t slice_offset,
    int64_t slice_size,
    bool add_inputs);
} // namespace

using punica_bgmv_shrink_fn = void (*)(
    at::Tensor& out,
    at::Tensor& input,
    at::Tensor& weights,
    at::Tensor& indicies,
    const double scale);

using punica_bgmv_expand_fn = void (*)(
    at::Tensor& out,
    at::Tensor& input,
    at::Tensor& weights,
    at::Tensor& indicies,
    bool add_inputs);

using punica_bgmv_expand_slice_fn = void (*)(
    at::Tensor& out,
    at::Tensor& input,
    at::Tensor& weights,
    at::Tensor& indicies,
    int64_t slice_offset,
    int64_t slice_size,
    bool add_inputs);

IPEX_DECLARE_DISPATCH(punica_bgmv_shrink_fn, punica_bgmv_shrink_kernel_stub);

IPEX_DECLARE_DISPATCH(punica_bgmv_expand_fn, punica_bgmv_expand_kernel_stub);

IPEX_DECLARE_DISPATCH(
    punica_bgmv_expand_slice_fn,
    punica_bgmv_expand_slice_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
