#pragma once

#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <array>
#include "ContextConvTranspose.h"
#include "OpContext.h"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace conv_transpose {

c10::intrusive_ptr<ConvTransposeOpContext> createConvTransposePrePackOpContext(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    int64_t groups,
    std::vector<int64_t>&& dilation,
    bool weight_is_channels_last,
    std::vector<int64_t>&& input_size);

at::Tensor conv_transpose_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvTransposeOpContext>& op_context);

ContextConvTranspose create(
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef output_padding,
    const at::IntArrayRef dilation,
    const int64_t groups,
    const bool weight_is_channels_last,
    const at::IntArrayRef input_size);

at::Tensor run(
    const ContextConvTranspose& context,
    const at::Tensor& input,
    const ideep::attr_t& attr);

// Runing backward for ConvTranspose by given grad_output, input and grad_masks.
// Will using the mkldnn_weight stored in the context
std::tuple<at::Tensor, at::Tensor, at::Tensor> run_backward(
    ContextConvTranspose& context,
    const at::Tensor& input,
    const at::Tensor& grad_output,
    std::array<bool, 3> output_mask);

// Return the n-D ATen weight which sharing same memory with the mkldnn packed
// weight This n-D ATen weight will be used for autograd and optimizer update
at::Tensor get_at_packed_weight(ContextConvTranspose& context);

// Pack given tensor to same format with mkldnn packed weight
at::Tensor pack(ContextConvTranspose& context, const at::Tensor& tensor);

// Unpack given tensor to same format with original weight format
at::Tensor unpack(ContextConvTranspose& context, const at::Tensor& tensor);

// query best weight format by given input size, and re-pack the mkldnn weight
// to newly queried format
void repack_for(ContextConvTranspose& context, std::vector<int64_t> input_size);

} // namespace conv_transpose
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
