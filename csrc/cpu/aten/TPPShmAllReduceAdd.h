#pragma once
#include <ATen/ATen.h>
#include <dyndisp/DispatchStub.h>
#include <torch/csrc/distributed/c10d/comm.hpp>

namespace torch_ipex {
namespace cpu {

void tpp_shmallreduce_forward(
    at::Tensor t_in,
    c10::intrusive_ptr<c10d::ProcessGroup> process_group);

using tpp_allreduce_impl_fn =
    void (*)(at::Tensor, c10::intrusive_ptr<c10d::ProcessGroup>);

IPEX_DECLARE_DISPATCH(tpp_allreduce_impl_fn, tpp_allreduce_kernel_stub);

} // namespace cpu
} // namespace torch_ipex