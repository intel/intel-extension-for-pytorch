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
    c10::optional<int64_t> batch_size);

at::Tensor linear_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context);

at::Tensor linear_relu_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context);

at::Tensor linear_gelu_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context,
    c10::string_view approximate);

at::Tensor linear_tanh_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context);

at::Tensor linear_sigmoid_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& op_context);

at::Tensor linear_swish_run(
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
    const c10::optional<int64_t> batch_size);

at::Tensor run(
    const ContextLinear& context,
    const at::Tensor& input,
    const ideep::attr_t& attr);

at::Tensor& run(
    const ContextLinear& context,
    const at::Tensor& input,
    at::Tensor& accumu,
    const ideep::attr_t& attr);

// Runing backward for ConvTranspose by given grad_output, input and grad_masks.
// Will using the mkldnn_weight stored in the context
std::tuple<at::Tensor, at::Tensor, at::Tensor> run_backward(
    ContextLinear& context,
    const at::Tensor& input,
    const at::Tensor& grad_output,
    std::array<bool, 3> output_mask);

// Pack given tensor to same format with mkldnn packed weight
at::Tensor pack(ContextLinear& context, const at::Tensor& tensor);

// Unpack given tensor to same format with original weight format
at::Tensor unpack(ContextLinear& context, const at::Tensor& tensor);

// query best weight format by given input size, and re-pack the mkldnn weight
// to newly queried format
void repack_for(ContextLinear& context, int64_t batch_size);

} // namespace linear
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
