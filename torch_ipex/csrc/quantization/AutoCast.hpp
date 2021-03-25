#pragma once

#include <ATen/Tensor.h>

namespace torch_ipex {
namespace autocast {
namespace int8 {

void clear_autocast_cache_int8();

at::Tensor conv2d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);

at::Tensor _convolution(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32);

at::Tensor _convolution(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    bool benchmark, bool deterministic, bool cudnn_enabled);

at::Tensor batch_norm(const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& running_mean, 
    const c10::optional<at::Tensor>& running_var, bool training, double momentum, double eps, bool cudnn_enabled);

at::Tensor max_pool2d(const at::Tensor& input, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode);

at::Tensor adaptive_avg_pool2d(const at::Tensor& input, at::IntArrayRef output_size);

at::Tensor relu(const at::Tensor& input);

at::Tensor& relu_(at::Tensor& input);

at::Tensor sigmoid(const at::Tensor& input);

at::Tensor linear(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias);

} // namespace int8
} // namespace autocast
} // namespace torch_ipex
