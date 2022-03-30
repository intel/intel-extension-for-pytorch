#pragma once

#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <array>
#include "ContextConvolution.h"
#include "OpContext.h"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace convolution {

c10::intrusive_ptr<ConvolutionOpContext> createConvolutionPrePackOpContext(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    std::vector<int64_t>&& kernel_size,
    int64_t groups,
    int64_t output_channel,
    bool weight_is_channels_last,
    std::vector<int64_t>&& input_size);

at::Tensor convolution_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context);

at::Tensor convolution_relu_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context);

at::Tensor convolution_leaky_relu_run(
    const at::Tensor& input,
    at::Scalar alpha,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context);

at::Tensor convolution_sigmoid_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context);

at::Tensor convolution_hardtanh_run(
    const at::Tensor& input,
    at::Scalar lower_bound,
    at::Scalar upper_bound,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context);

at::Tensor convolution_elu_run(
    const at::Tensor& input,
    at::Scalar alpha,
    at::Scalar scale,
    at::Scalar input_scale,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context);

at::Tensor convolution_swish_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context);

at::Tensor convolution_add_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context);

at::Tensor convolution_add_relu_run(
    const at::Tensor& input,
    at::Tensor& accumu,
    const c10::optional<at::Scalar>& alpha,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context);

at::Tensor& convolution_bottleneck_run(
    at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context1,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context2,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context3);

at::Tensor convolution_bottleneck_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context1,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context2,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context3,
    const c10::intrusive_ptr<ConvolutionOpContext>& op_context4);

ContextConvolution create(
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    const at::IntArrayRef kerel_size,
    const int64_t groups,
    const int64_t output_channel,
    const bool weight_is_channels_last,
    const std::vector<int64_t>& input_size,
    const ideep::attr_t& attr);

at::Tensor run(
    const ContextConvolution& context,
    const at::Tensor& input,
    const ideep::attr_t& attr);

at::Tensor& run(
    const ContextConvolution& context,
    const at::Tensor& input,
    at::Tensor& accumu,
    const ideep::attr_t& attr);

// Runing backward for conv by given grad_output, input and grad_masks.
// Will using the mkldnn_weight/bias stored in the context
std::tuple<at::Tensor, at::Tensor, at::Tensor> run_backward(
    ContextConvolution& context,
    const at::Tensor& input,
    const at::Tensor& grad_output,
    std::array<bool, 3> output_mask);

// Return the n-D ATen weight which sharing same memory with the mkldnn packed
// weight This n-D ATen weight will be used for autograd and optimizer update
at::Tensor get_at_packed_weight(ContextConvolution& context);

// Pack given tensor to same format with mkldnn packed weight
at::Tensor pack(ContextConvolution& context, const at::Tensor& tensor);

// Unpack given tensor to same format with original weight format
at::Tensor unpack(ContextConvolution& context, const at::Tensor& tensor);

} // namespace convolution
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
