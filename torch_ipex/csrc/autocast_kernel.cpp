#include "autocast_kernel.hpp"
#include "autocast_mode.h"
#include "autocast_verbose.h"
#include "cpu/BatchNorm.h"
#include "quantization/AutoCast.hpp"

namespace torch_ipex {
namespace autocast {

template <class Ret, class F, class... Args>
Ret DataTypeCastFuction(F Quant, F At, std::string op_name, Args... args) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name(op_name);
#endif
  if (is_quantization_enabled()) {
    return Quant(cpu_cached_cast(target_type, args)...);
  } else {
    return At(cpu_cached_cast(target_type, args)...);
  }
}

template <class Ret, class F, class... Args>
Ret FallThroughFuction(F Quant, F At, std::string op_name, Args... args) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name(op_name);
#endif
  if (is_quantization_enabled()) {
    return Quant(args...);
  } else {
    return At(args...);
  }
}

template <class Ret, class F, class... Args>
Ret FP32CastFunction(F Quant, F At, std::string op_name, Args... args) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto at_target_type = at::kFloat;
  auto target_type = get_autocast_dtype();
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name(op_name);
#endif
  if (is_quantization_enabled()) {
    return Quant(cpu_cached_cast(target_type, args)...);
  } else {
    return At(cpu_cached_cast(at_target_type, args)...);
  }
}

at::Tensor conv2d(const at::Tensor &input, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
                  at::IntArrayRef padding, at::IntArrayRef dilation,
                  int64_t groups) {
  return DataTypeCastFuction<at::Tensor>(int8::conv2d, at::conv2d, "conv2d",
                                         input, weight, bias, stride, padding,
                                         dilation, groups);
}

at::Tensor conv3d(const at::Tensor &input, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
                  at::IntArrayRef padding, at::IntArrayRef dilation,
                  int64_t groups) {
  return DataTypeCastFuction<at::Tensor>(int8::conv3d, at::conv3d, "conv3d",
                                         input, weight, bias, stride, padding,
                                         dilation, groups);
}

at::Tensor conv_transpose3d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation) {
  return FP32CastFunction<at::Tensor>(
      int8::conv_transpose3d, at::conv_transpose3d, "conv_transpose3d", input,
      weight, bias, stride, padding, output_padding, groups, dilation);
}

at::Tensor _convolution(const at::Tensor& input, const at::Tensor& weight, const  c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("_convolution");
#endif
  if (is_quantization_enabled()) {
    return int8::_convolution(input, weight, bias, stride, padding, dilation,
                              transposed, output_padding, groups, benchmark,
                              deterministic, cudnn_enabled, allow_tf32);
  }
  // makeFallthrough to support 3DUNET transposed conv3d jit path
  return at::_convolution(input, weight, bias, stride, padding, dilation,
                          transposed, output_padding, groups, benchmark,
                          deterministic, cudnn_enabled, allow_tf32);
}

at::Tensor _convolution_deprecated(const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups,
    bool benchmark, bool deterministic, bool cudnn_enabled) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("_convolution_deprecated");
#endif
  auto target_type = get_autocast_dtype();
  if (is_quantization_enabled()) {
    return int8::_convolution(input, weight, bias, stride, padding, dilation, transposed, 
                              output_padding, groups, benchmark, deterministic, cudnn_enabled);
  }
  return at::_convolution(
      cpu_cached_cast(target_type, input), cpu_cached_cast(target_type, weight),
      cpu_cached_cast(target_type, bias), stride, padding, dilation, transposed,
      output_padding, groups, benchmark, deterministic, cudnn_enabled);
}

at::Tensor batch_norm(const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& running_mean, 
    const c10::optional<at::Tensor>& running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  return FallThroughFuction<at::Tensor>(
      int8::batch_norm, torch_ipex::cpu::batch_norm, "batch_norm", input,
      weight, bias, running_mean, running_var, training, momentum, eps,
      cudnn_enabled);
}

at::Tensor linear(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias) {
  return DataTypeCastFuction<at::Tensor>(int8::linear, at::linear, "linear",
                                         input, weight, bias);
}

at::Tensor max_pool2d(const at::Tensor& input, at::IntArrayRef kernel_size, at::IntArrayRef stride,
        at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  return FallThroughFuction<at::Tensor>(int8::max_pool2d, at::max_pool2d,
                                        "max_pool2d", input, kernel_size,
                                        stride, padding, dilation, ceil_mode);
}

at::Tensor adaptive_avg_pool2d(const at::Tensor& input, at::IntArrayRef output_size) {
  return FallThroughFuction<at::Tensor>(
      int8::adaptive_avg_pool2d, at::adaptive_avg_pool2d, "adaptive_avg_pool2d",
      input, output_size);
}

at::Tensor relu(const at::Tensor& input) {
  return FallThroughFuction<at::Tensor>(int8::relu, at::relu, "relu", input);
}

at::Tensor& relu_(at::Tensor& input) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("relu_");
#endif
  if (is_quantization_enabled()) {
    return int8::relu_(input);
  }
  return at::relu_(input);
}

at::Tensor sigmoid(const at::Tensor& input) {
  return FallThroughFuction<at::Tensor>(int8::sigmoid, at::sigmoid, "sigmoid",
                                        input);
}

at::Tensor& add_tensor_(at::Tensor& input, const at::Tensor& other, const at::Scalar& alpha) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("add_tensor_");
#endif
  if (is_quantization_enabled()) {
    return int8::add_tensor_(input, other, alpha);
  }
  // make fall makeFallthrough.
  input.add_(other, alpha);
  return input;
}

at::Tensor add_tensor(const at::Tensor& input, const at::Tensor& other, const at::Scalar& alpha) {
  return FallThroughFuction<at::Tensor>(int8::add_tensor, at::add, "add_tensor",
                                        input, other, alpha);
}

at::Tensor dropout(const at::Tensor& input, double p, bool train) {
  return FallThroughFuction<at::Tensor>(int8::dropout, at::dropout, "dropout",
                                        input, p, train);
}

at::Tensor gelu(const at::Tensor& input) {
  return FallThroughFuction<at::Tensor>(int8::gelu, at::gelu, "gelu", input);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
lstm_aten(const at::Tensor &_input, at::TensorList hx, at::TensorList _params,
          bool has_biases, int64_t num_layers, double dropout_p, bool train,
          bool bidirectional, bool batch_first) {
  return FallThroughFuction<std::tuple<at::Tensor, at::Tensor, at::Tensor>>(
      int8::lstm, at::lstm, "lstm", _input, hx, _params, has_biases, num_layers,
      dropout_p, train, bidirectional, batch_first);
}

at::Tensor flatten(const at::Tensor &input, int64_t start_dim,
                   int64_t end_dim) {
  return FallThroughFuction<at::Tensor>(int8::flatten, at::flatten, "flatten",
                                        input, start_dim, end_dim);
}

at::Tensor matmul(const at::Tensor& mat1, const at::Tensor& mat2) {
  return DataTypeCastFuction<at::Tensor>(
      int8::matmul, at::matmul, "matmul", mat1, mat2);
}

} // autocast
} // torch_ipex
