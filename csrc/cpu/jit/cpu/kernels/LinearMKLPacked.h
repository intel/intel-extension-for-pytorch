#pragma once

#include <ATen/Tensor.h>
#include "ContextLinearMKL.h"
#include "OpContext.h"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace mkl_sgemm {

c10::intrusive_ptr<MKLOpContext> createLinearMKLPrePackOpContext(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    c10::optional<int64_t> batch_size);

at::Tensor mkl_sgemm_run(
    const at::Tensor& input,
    c10::intrusive_ptr<MKLOpContext> op_context);

ContextLinearMKL create(
    at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<int64_t> batch_size);

at::Tensor run(ContextLinearMKL& context, const at::Tensor& input);

at::Tensor& run(
    ContextLinearMKL& context,
    const at::Tensor& input,
    at::Tensor& accumu);

at::Tensor pack(ContextLinearMKL& context, const at::Tensor& tensor);

} // namespace mkl_sgemm
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
