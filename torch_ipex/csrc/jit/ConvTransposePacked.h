#pragma once

#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <array>
#include "ContextConvTranspose.h"
#include "OpContext.h"

namespace torch_ipex {
namespace cpu {
namespace detail {
namespace conv_transpose2d {

c10::intrusive_ptr<ConvTransposeOpContext> createConvTransposePrePackOpContext(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    int64_t groups,
    std::vector<int64_t>&& dilation,
    std::vector<int64_t>&& kernel_size,
    int64_t output_channel,
    bool weight_is_channels_last,
    bool weight_is_packed,
    std::vector<int64_t>&& input_size);

at::Tensor conv_transpose2d_run(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvTransposeOpContext>& op_context);

ContextConvTranspose create(
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef output_padding,
    const at::IntArrayRef dilation,
    const at::IntArrayRef kerel_size,
    const int64_t groups,
    const int64_t output_channel,
    const bool weight_is_channels_last,
    const bool weight_is_packed,
    const at::IntArrayRef input_size);

at::Tensor run(
    const ContextConvTranspose& context,
    const at::Tensor& input,
    const ideep::attr_t& attr);

} // namespace conv_transpose2d
} // namespace detail
} // namespace cpu
} // namespace torch_ipex
