#pragma once

#include <ATen/Tensor.h>
#include "ContextLinear.h"
#include "OpContext.h"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace linear {

c10::intrusive_ptr<LinearOpContext> createLinearPrePackOpContext(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    int64_t out_features,
    int64_t in_features,
    int64_t batch_size,
    bool weight_is_packed);

at::Tensor linear_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context);

at::Tensor linear_relu_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context);

at::Tensor linear_gelu_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context);

at::Tensor linear_add_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const c10::intrusive_ptr<LinearOpContext>& op_context);

ContextLinear create(
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const int64_t out_features,
    const int64_t in_features,
    const int64_t batch_size,
    const bool weight_is_packed);

at::Tensor run(
    const ContextLinear& context,
    const at::Tensor& input,
    const ideep::attr_t& attr);

at::Tensor& run(
    const ContextLinear& context,
    const at::Tensor& input,
    at::Tensor& accumu,
    const ideep::attr_t& attr);

} // namespace linear
} // namespace detail
} // namespace cpu
} // namespace torch_ipex