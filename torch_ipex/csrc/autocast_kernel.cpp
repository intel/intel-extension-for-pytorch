#include "autocast_mode.h"
#include "autocast_kernel.hpp"

#include "quantization/AutoCast.hpp"

namespace torch_ipex {
namespace autocast {

at::Tensor conv2d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);

  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::conv2d(input, weight, bias, stride, padding, dilation, groups);
  }
  return at::conv2d(cpu_cached_cast(target_type, input),
                    cpu_cached_cast(target_type, weight),
                    cpu_cached_cast(target_type, bias),
                    stride, padding, dilation, groups);
}

at::Tensor _convolution(const at::Tensor& input, const at::Tensor& weight, const  c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::_convolution(input, weight, bias, stride, padding, dilation, transposed,
                              output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
  }
  return at::_convolution(cpu_cached_cast(target_type, input),
                          cpu_cached_cast(target_type, weight),
                          cpu_cached_cast(target_type, bias),
                          stride, padding, dilation, transposed, output_padding,
                          groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
}

at::Tensor _convolution_deprecated(const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups,
    bool benchmark, bool deterministic, bool cudnn_enabled) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::_convolution(input, weight, bias, stride, padding, dilation, transposed, 
                              output_padding, groups, benchmark, deterministic, cudnn_enabled);
  }
  return at::_convolution(cpu_cached_cast(target_type, input),
                            cpu_cached_cast(target_type, weight),
                            cpu_cached_cast(target_type, bias),
                            stride, padding, dilation, transposed, output_padding,
                            groups, benchmark, deterministic, cudnn_enabled); 
}

at::Tensor batch_norm(const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& running_mean, 
    const c10::optional<at::Tensor>& running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  }
  // convert to fp32 path.
  return at::batch_norm(cpu_cached_cast(at::kFloat, input),
                        cpu_cached_cast(at::kFloat, weight),
                        cpu_cached_cast(at::kFloat, bias),
                        cpu_cached_cast(at::kFloat, running_mean),
                        cpu_cached_cast(at::kFloat, running_var),
                        training, momentum, eps, cudnn_enabled);
}

at::Tensor linear(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::linear(input, weight, bias);
  }
 
  return at::linear(cpu_cached_cast(target_type, input),
                    cpu_cached_cast(target_type, weight),
                    cpu_cached_cast(target_type, bias));
}

at::Tensor max_pool2d(const at::Tensor& input, at::IntArrayRef kernel_size, at::IntArrayRef stride,
        at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode);
  }
  // convert to fp32 path.
  return at::max_pool2d(cpu_cached_cast(at::kFloat, input), kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor adaptive_avg_pool2d(const at::Tensor& input, at::IntArrayRef output_size) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::adaptive_avg_pool2d(input, output_size);
  }
  //convert to fp32 path..
  return at::adaptive_avg_pool2d(cpu_cached_cast(at::kFloat, input), output_size);
}

at::Tensor relu(const at::Tensor& input) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::relu(input);
  }
  // make fall makeFallthrough.
  return at::relu(input);
}

} // autocast
} // torch_ipex
