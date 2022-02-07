#pragma once

#include <ATen/Tensor.h>
#include <csrc/dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

#if defined(DYN_DISP_BUILD)
namespace {
#endif

void rnnt_embedding_kernel_impl(
    const at::Tensor& embedding_table,
    const at::Tensor& idx,
    at::Tensor embedding_out,
    int64_t _SOS,
    int64_t batch_size,
    int64_t embedding_dim);

#if defined(DYN_DISP_BUILD)
}
#endif

using rnnt_embedding_kernel_fn = void (*)(
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor,
    int64_t,
    int64_t,
    int64_t);
DECLARE_DISPATCH(rnnt_embedding_kernel_fn, rnnt_embedding_kernel_stub);

} // namespace cpu
} // namespace torch_ipex