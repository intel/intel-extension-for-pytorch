#pragma once

#include <ATen/ATen.h>

#include <dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

at::Tensor& index_select_out_cpu_(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    at::Tensor& result);

at::Tensor index_select_cpu_(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index);

namespace {

void index_select_contig_kernel(
    const at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index);

void copy_kernel(at::TensorIterator& iter, bool /*non_blocking*/);

} // namespace

using index_select_fn =
    void (*)(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&);
IPEX_DECLARE_DISPATCH(index_select_fn, index_select_contig_stub);

using copy_fn = void (*)(at::TensorIterator&, bool non_blocking);
IPEX_DECLARE_DISPATCH(copy_fn, copy_stub);

} // namespace cpu
} // namespace torch_ipex