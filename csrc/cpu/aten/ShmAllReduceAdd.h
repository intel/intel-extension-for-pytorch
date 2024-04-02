#pragma once

#ifdef USE_CCL
#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

namespace {

at::Tensor shm_all_reduce_add(
    at::Tensor& t_in,
    at::Tensor& t_address,
    at::Tensor& t_state,
    at::Tensor& t_blockState,
    int64_t shm_block_size,
    int64_t rank,
    int64_t world_size);
}

using shm_all_reduce_add_kernel_fn = at::Tensor (*)(
    at::Tensor& t_in,
    at::Tensor& t_address,
    at::Tensor& t_state,
    at::Tensor& t_blockState,
    int64_t shm_block_size,
    int64_t rank,
    int64_t world_size);

IPEX_DECLARE_DISPATCH(
    shm_all_reduce_add_kernel_fn,
    shm_all_reduce_add_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
#endif