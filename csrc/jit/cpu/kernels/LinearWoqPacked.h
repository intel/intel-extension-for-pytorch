#pragma once

#include <ATen/Tensor.h>
#include "ContextLinearWoq.h"
#include "OpContext.h"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace woq_linear {

// WOQ = weight-only quantization
c10::intrusive_ptr<WoqLinearOpContext> createWoqLinearPrePackOpContext(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    c10::optional<int64_t> batch_size);

at::Tensor woq_linear_run(
    const at::Tensor& input,
    c10::intrusive_ptr<WoqLinearOpContext> op_context);

ContextLinearWoq create(
    at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<int64_t> batch_size);

at::Tensor run(ContextLinearWoq& context, const at::Tensor& input);

at::Tensor pack(ContextLinearWoq& context, const at::Tensor& tensor);

} // namespace woq_linear
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
