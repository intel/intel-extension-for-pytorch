#pragma once

#include <ATen/Tensor.h>
#include <csrc/dyndisp/DispatchStub.h>

namespace torch_ipex {
namespace cpu {

#if defined(DYN_DISP_BUILD)
namespace {
#endif
void sum_kernel_impl(at::TensorIterator& iter);

#if defined(DYN_DISP_BUILD)
}
#endif

using sum_kernel_fn = void (*)(at::TensorIterator&);
DECLARE_DISPATCH(sum_kernel_fn, sum_kernel_stub);

at::Tensor sum_out_cpu(
    const at::Tensor& input,
    c10::IntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype);

} // namespace cpu
} // namespace torch_ipex
