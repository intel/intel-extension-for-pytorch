#pragma once

#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <csrc/dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

at::Tensor& cat_out_cpu(at::TensorList tensors, int64_t dim, at::Tensor& out);

at::Tensor cat_cpu(at::TensorList tensors, int64_t dim);

namespace {

void cat_contig_kernel(
    const at::Tensor& result,
    const at::MaterializedITensorListRef& tensors,
    int64_t dim,
    bool all_same_sizes_and_stride);

} // namespace

using cat_contig_fn = void (*)(
    const at::Tensor&,
    const at::MaterializedITensorListRef&,
    int64_t,
    bool);
DECLARE_DISPATCH(cat_contig_fn, cat_contig_stub);

} // namespace cpu
} // namespace torch_ipex