#include "autocast_kernel.hpp"
#include "autocast_mode.h"
#include "csrc/aten/cpu/BatchNorm.h"
#include "csrc/quantization/AutoCast.hpp"

namespace torch_ipex {
namespace autocast {

template <class Ret, class F, class... Args>
Ret DataTypeCastFuction(
    F Quant,
    F At,
    std::string register_op_name,
    Args... args) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
  if (is_quantization_enabled()) {
    return Quant(cpu_cached_cast(target_type, args)...);
  } else {
    return At(cpu_cached_cast(target_type, args)...);
  }
}

template <class Ret, class F, class... Args>
Ret FallThroughFuction(
    F Quant,
    F At,
    std::string register_op_name,
    Args... args) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  if (is_quantization_enabled()) {
    return Quant(args...);
  } else {
    return At(args...);
  }
}

template <class Ret, class F, class... Args>
Ret FP32CastFunction(
    F Quant,
    F At,
    std::string register_op_name,
    Args... args) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  if (is_quantization_enabled()) {
    return Quant(cpu_cached_cast(at::kFloat, args)...);
  } else {
    return At(cpu_cached_cast(at::kFloat, args)...);
  }
}

at::Tensor conv2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  return DataTypeCastFuction<at::Tensor>(
      int8::conv2d,
      at::conv2d,
      "conv2d",
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups);
}

at::Tensor conv3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  return DataTypeCastFuction<at::Tensor>(
      int8::conv3d,
      at::conv3d,
      "conv3d",
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups);
}

at::Tensor conv_transpose3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation) {
  return FP32CastFunction<at::Tensor>(
      int8::conv_transpose3d,
      at::conv_transpose3d,
      "conv_transpose3d",
      input,
      weight,
      bias,
      stride,
      padding,
      output_padding,
      groups,
      dilation);
}

at::Tensor _convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  if (is_quantization_enabled()) {
    return int8::_convolution(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        benchmark,
        deterministic,
        cudnn_enabled,
        allow_tf32);
  }
  // makeFallthrough to support 3DUNET transposed conv3d jit path
  return at::_convolution(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      benchmark,
      deterministic,
      cudnn_enabled,
      allow_tf32);
}

at::Tensor _convolution_deprecated(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  auto target_type = get_autocast_dtype();
  if (is_quantization_enabled()) {
    return int8::_convolution(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        benchmark,
        deterministic,
        cudnn_enabled);
  }
  return at::_convolution(
      cpu_cached_cast(target_type, input),
      cpu_cached_cast(target_type, weight),
      cpu_cached_cast(target_type, bias),
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      benchmark,
      deterministic,
      cudnn_enabled);
}

at::Tensor batch_norm(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    bool training,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  return FallThroughFuction<at::Tensor>(
      int8::batch_norm,
      torch_ipex::cpu::batch_norm,
      "batch_norm",
      input,
      weight,
      bias,
      running_mean,
      running_var,
      training,
      momentum,
      eps,
      cudnn_enabled);
}

at::Tensor linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias) {
  return DataTypeCastFuction<at::Tensor>(
      int8::linear, at::linear, "linear", input, weight, bias);
}

at::Tensor max_pool2d(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  return FallThroughFuction<at::Tensor>(
      int8::max_pool2d,
      at::max_pool2d,
      "max_pool2d",
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
}

at::Tensor adaptive_avg_pool2d(
    const at::Tensor& input,
    at::IntArrayRef output_size) {
  return FallThroughFuction<at::Tensor>(
      int8::adaptive_avg_pool2d,
      at::adaptive_avg_pool2d,
      "adaptive_avg_pool2d",
      input,
      output_size);
}

at::Tensor avg_pool2d(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::avg_pool2d(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
}

std::tuple<at::Tensor, at::Tensor> adaptive_max_pool2d(
    const at::Tensor& input,
    at::IntArrayRef output_size) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::adaptive_max_pool2d(input, output_size);
}

at::Tensor upsample_nearest1d(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::upsample_nearest1d(input, output_size, scales);
}

at::Tensor upsample_nearest1d_vec(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::upsample_nearest1d(input, output_size, scale_factors);
}

at::Tensor upsample_nearest2d(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::upsample_nearest2d(input, output_size, scales_h, scales_w);
}

at::Tensor upsample_nearest2d_vec(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::upsample_nearest2d(input, output_size, scale_factors);
}

at::Tensor upsample_nearest3d(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::upsample_nearest3d(
      input, output_size, scales_d, scales_h, scales_w);
}

at::Tensor upsample_nearest3d_vec(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::upsample_nearest3d(input, output_size, scale_factors);
}

at::Tensor upsample_linear1d(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::upsample_linear1d(input, output_size, align_corners, scales);
}

at::Tensor upsample_linear1d_vec(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::upsample_linear1d(
      input, output_size, align_corners, scale_factors);
}

at::Tensor upsample_bilinear2d(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::upsample_bilinear2d(
      input, output_size, align_corners, scales_h, scales_w);
}

at::Tensor upsample_bilinear2d_vec(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::upsample_bilinear2d(
      input, output_size, align_corners, scale_factors);
}

at::Tensor upsample_trilinear3d(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::upsample_trilinear3d(
      input, output_size, align_corners, scales_d, scales_h, scales_w);
}

at::Tensor upsample_trilinear3d_vec(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::upsample_trilinear3d(
      input, output_size, align_corners, scale_factors);
}

at::Tensor relu(const at::Tensor& input) {
  return FallThroughFuction<at::Tensor>(int8::relu, at::relu, "relu", input);
}

at::Tensor& relu_(at::Tensor& input) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  if (is_quantization_enabled()) {
    return int8::relu_(input);
  }
  return at::relu_(input);
}

at::Tensor sigmoid(const at::Tensor& input) {
  return FallThroughFuction<at::Tensor>(
      int8::sigmoid, at::sigmoid, "sigmoid", input);
}

at::Tensor& add_tensor_(
    at::Tensor& input,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  if (is_quantization_enabled()) {
    return int8::add_tensor_(input, other, alpha);
  }
  // make fall makeFallthrough.
  input.add_(other, alpha);
  return input;
}

at::Tensor add_tensor(
    const at::Tensor& input,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  return FallThroughFuction<at::Tensor>(
      int8::add_tensor, at::add, "add_tensor", input, other, alpha);
}

at::Tensor dropout(const at::Tensor& input, double p, bool train) {
  return FallThroughFuction<at::Tensor>(
      int8::dropout, at::dropout, "dropout", input, p, train);
}

at::Tensor gelu(const at::Tensor& input) {
  return FallThroughFuction<at::Tensor>(int8::gelu, at::gelu, "gelu", input);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_aten(
    const at::Tensor& _input,
    at::TensorList hx,
    at::TensorList _params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first) {
  return FallThroughFuction<std::tuple<at::Tensor, at::Tensor, at::Tensor>>(
      int8::lstm,
      at::lstm,
      "lstm",
      _input,
      hx,
      _params,
      has_biases,
      num_layers,
      dropout_p,
      train,
      bidirectional,
      batch_first);
}

at::Tensor flatten(
    const at::Tensor& input,
    int64_t start_dim,
    int64_t end_dim) {
  return FallThroughFuction<at::Tensor>(
      int8::flatten, at::flatten, "flatten", input, start_dim, end_dim);
}

at::Tensor matmul(const at::Tensor& mat1, const at::Tensor& mat2) {
  return DataTypeCastFuction<at::Tensor>(
      int8::matmul, at::matmul, "matmul", mat1, mat2);
}

at::Tensor avg_pool1d(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::avg_pool1d(
      input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

at::Tensor binary_cross_entropy_with_logits(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& pos_weight,
    int64_t reduction) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::binary_cross_entropy_with_logits(
      self, target, weight, pos_weight, reduction);
}

at::Tensor searchsorted_tensor(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    c10::optional<c10::string_view> side,
    const c10::optional<at::Tensor>& sorter) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::searchsorted(
      sorted_sequence, self, out_int32, right, side, sorter);
}

at::Tensor searchsorted_scalar(
    const at::Tensor& sorted_sequence,
    const at::Scalar& self,
    bool out_int32,
    bool right,
    c10::optional<c10::string_view> side,
    const c10::optional<at::Tensor>& sorter) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::searchsorted(
      sorted_sequence, self, out_int32, right, side, sorter);
}

at::Tensor tril(const at::Tensor& self, int64_t diagonal) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::tril(self, diagonal);
}

at::Tensor triu(const at::Tensor& self, int64_t diagonal) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::triu(self, diagonal);
}

at::Tensor dot(const at::Tensor& self, const at::Tensor& tensor) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::dot(self, tensor);
}

at::Tensor vdot(const at::Tensor& self, const at::Tensor& other) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::vdot(self, other);
}

at::Tensor im2col(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::im2col(self, kernel_size, dilation, padding, stride);
}

at::Tensor col2im(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::col2im(self, output_size, kernel_size, dilation, padding, stride);
}

::std::tuple<at::Tensor, at::Tensor> cummax(
    const at::Tensor& self,
    int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::cummax(self, dim);
}

::std::tuple<at::Tensor, at::Tensor> cummax_dimname(
    const at::Tensor& self,
    at::Dimname dim) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::cummax(self, dim);
}

::std::tuple<at::Tensor, at::Tensor> cummin(
    const at::Tensor& self,
    int64_t dim) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::cummin(self, dim);
}

::std::tuple<at::Tensor, at::Tensor> cummin_dimname(
    const at::Tensor& self,
    at::Dimname dim) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::cummin(self, dim);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor> lu_unpack(
    const at::Tensor& LU_data,
    const at::Tensor& LU_pivots,
    bool unpack_data,
    bool unpack_pivots) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots);
}

::std::tuple<at::Tensor, at::Tensor> adaptive_max_pool1d(
    const at::Tensor& self,
    at::IntArrayRef output_size) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  return at::adaptive_max_pool1d(self, output_size);
}

} // namespace autocast
} // namespace torch_ipex
