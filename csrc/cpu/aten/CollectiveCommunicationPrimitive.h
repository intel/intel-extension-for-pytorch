#pragma once
#ifdef USE_CCL
#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

namespace {

at::Tensor all_reduce_add(at::Tensor& t_in);
at::Tensor allgather(
    at::Tensor t_in,
    std::vector<int64_t> cols_per_rank,
    int64_t world_size);
int64_t get_world_size(const at::Tensor dummy_input);
int64_t get_rank(const at::Tensor dummy_input);
} // namespace

using all_reduce_add_fn = at::Tensor (*)(at::Tensor& t_in);
using allgather_fn = at::Tensor (*)(
    at::Tensor t_in,
    std::vector<int64_t> cols_per_rank,
    int64_t world_size);

IPEX_DECLARE_DISPATCH(all_reduce_add_fn, all_reduce_add_kernel_stub);
IPEX_DECLARE_DISPATCH(allgather_fn, allgather_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
#endif