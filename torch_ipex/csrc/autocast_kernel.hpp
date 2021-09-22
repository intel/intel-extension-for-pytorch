#pragma once

#include <ATen/ATen.h>

namespace torch_ipex {
namespace autocast {

at::Tensor conv2d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);

at::Tensor conv3d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);

at::Tensor conv_transpose3d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation);

at::Tensor _convolution(const at::Tensor& input, const at::Tensor& weight, const  c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32);

at::Tensor _convolution_deprecated(const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups,
    bool benchmark, bool deterministic, bool cudnn_enabled);

at::Tensor batch_norm(const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& running_mean, 
    const c10::optional<at::Tensor>& running_var, bool training, double momentum, double eps, bool cudnn_enabled);

at::Tensor linear(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias);

at::Tensor max_pool2d(const at::Tensor& input, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode);

at::Tensor adaptive_avg_pool2d(const at::Tensor& input, at::IntArrayRef output_size);

at::Tensor relu(const at::Tensor& input);

at::Tensor& relu_(at::Tensor& input);

at::Tensor sigmoid(const at::Tensor& input);

at::Tensor& add_tensor_(at::Tensor& input, const at::Tensor& other, const at::Scalar& alpha);

at::Tensor add_tensor(const at::Tensor& input, const at::Tensor& other, const at::Scalar& alpha);

at::Tensor dropout(const at::Tensor& input, double p, bool train);

at::Tensor gelu(const at::Tensor& input);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
lstm_aten(const at::Tensor &_input, at::TensorList hx, at::TensorList _params,
          bool has_biases, int64_t num_layers, double dropout_p, bool train,
          bool bidirectional, bool batch_first);

at::Tensor flatten(const at::Tensor &input, int64_t start_dim, int64_t end_dim);

at::Tensor matmul(const at::Tensor& mat1, const at::Tensor& mat2);

} // autocast
} // torch_ipex
