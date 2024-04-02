#pragma once

#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
rotary_position_embedding_kernel_impl(
    at::Tensor& t_in,
    at::Tensor& t_emb_pos,
    at::Tensor& t_pos,
    int64_t N, // N: number of head, H: head size
    int64_t H,
    int64_t offset,
    int64_t rotary_ndims);
}

using rotary_position_embedding_kernel_fn =
    std::tuple<at::Tensor, at::Tensor, at::Tensor> (*)(
        at::Tensor& t_in,
        at::Tensor& t_emb_pos,
        at::Tensor& t_pos,
        int64_t N, // N: number of head, H: head size
        int64_t H,
        int64_t offset,
        int64_t rotary_ndims);

IPEX_DECLARE_DISPATCH(
    rotary_position_embedding_kernel_fn,
    rotary_position_embedding_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
