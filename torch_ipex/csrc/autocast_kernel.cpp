#include "autocast_mode.h"
#include "autocast_kernel.hpp"
#include "autocast_verbose.h"
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
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("conv2d");
#endif
  return at::conv2d(cpu_cached_cast(target_type, input),
                    cpu_cached_cast(target_type, weight),
                    cpu_cached_cast(target_type, bias),
                    stride, padding, dilation, groups);
}

at::Tensor conv3d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);

  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::conv3d(input, weight, bias, stride, padding, dilation, groups);
  }
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("conv3d");
#endif
  return at::conv3d(cpu_cached_cast(target_type, input),
                    cpu_cached_cast(target_type, weight),
                    cpu_cached_cast(target_type, bias),
                    stride, padding, dilation, groups);
}

at::Tensor conv_transpose3d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);

  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
  }
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("conv_transpose3d");
#endif
  return at::conv_transpose3d(cpu_cached_cast(at::kFloat, input),
                              cpu_cached_cast(at::kFloat, weight),
                              cpu_cached_cast(at::kFloat, bias),
                              stride, padding, output_padding, groups, dilation);
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
  // makeFallthrough to support 3DUNET transposed conv3d jit path
  return at::_convolution(input, weight, bias, stride, padding, dilation, transposed,
                          output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
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
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("_convolution_deprecated");
#endif
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
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("batch_norm");
#endif
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
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("linear");
#endif
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
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("max_pool2d");
#endif
  return at::max_pool2d(cpu_cached_cast(at::kFloat, input), kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor adaptive_avg_pool2d(const at::Tensor& input, at::IntArrayRef output_size) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::adaptive_avg_pool2d(input, output_size);
  }
  //convert to fp32 path.
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("adaptive_avg_pool2d");
#endif
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

at::Tensor& relu_(at::Tensor& input) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::relu_(input);
  }
  // make fall makeFallthrough.
  return at::relu_(input);
}

at::Tensor sigmoid(const at::Tensor& input) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::sigmoid(input);
  }
  // make fall makeFallthrough.
  return at::sigmoid(input);
}

at::Tensor& add_tensor_(at::Tensor& input, const at::Tensor& other, const at::Scalar& alpha) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::add_tensor_(input, other, alpha);
  }
  // make fall makeFallthrough.
  input.add_(other, alpha);
  return input;
}

at::Tensor add_tensor(const at::Tensor& input, const at::Tensor& other, const at::Scalar& alpha) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::add_tensor(input, other, alpha);
  }
  // make fall makeFallthrough.
  return at::add(input, other, alpha);
}

at::Tensor dropout(const at::Tensor& input, double p, bool train) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
  if (at::ScalarType::Char == target_type) {
    return int8::dropout(input, p, train);
  }
  //convert to fp32 path.
#if defined(ENABLE_AUTOCAST_VERBOSE)
  verbose::OpNameGuard op_name("dropout");
#endif
  return at::dropout(cpu_cached_cast(at::kFloat, input), p, train);
}

} // autocast
} // torch_ipex
