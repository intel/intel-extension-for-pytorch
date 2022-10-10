#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NativeFunctions.h>
#include <ATen/record_function.h>
#include <grp.h>
#include <intrinsic/intrinsic.h>
#include <oneapi/dnnl/dnnl.hpp>

#include <oneDNN/oneDNN.h>

using namespace xpu::oneDNN;

namespace torch {
namespace jit {
namespace xpu {

at::Tensor& conv2d_sum(
    at::Tensor& accumu,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    at::Scalar alpha) {
  RECORD_FUNCTION(
      "conv2d_sum", std::vector<c10::IValue>({input, weight, bias, accumu}));
  const OptionalDeviceGuard device_guard(device_of(input));
  at::AtenIpexTypeXPU::convolution_sum(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      false,
      {{0, 0}},
      groups,
      accumu,
      alpha);
  return accumu;
}

at::Tensor& conv2d_sum_relu(
    at::Tensor& accumu,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    at::Scalar alpha) {
  RECORD_FUNCTION(
      "conv2d_sum_relu",
      std::vector<c10::IValue>({input, weight, bias, accumu}));
  const OptionalDeviceGuard device_guard(device_of(input));
  at::AtenIpexTypeXPU::convolution_sum_relu(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      false,
      {{0, 0}},
      groups,
      accumu,
      alpha);
  return accumu;
}

at::Tensor q_conv2d_dequantize(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point) {
  RECORD_FUNCTION(
      "q_conv2d_dequantize", std::vector<c10::IValue>({input, packed_weight}));
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::q_conv2d_dequantize(
      input, packed_weight, output_scale, output_zero_point);
}

at::Tensor softplus_tanh(
    const Tensor& self,
    const Scalar& beta,
    const Scalar& threshold) {
  const OptionalDeviceGuard device_guard(device_of(self));
  auto _self = to_plain_if_needed(self);
  return at::AtenIpexTypeXPU::softplus_tanh(_self, beta, threshold);
}

at::Tensor softplus_tanh_mul(
    const Tensor& self,
    const Scalar& beta,
    const Scalar& threshold,
    const Tensor& mul_input) {
  const OptionalDeviceGuard device_guard(device_of(self));
  auto _self = to_plain_if_needed(self);
  auto _mul = to_plain_if_needed(mul_input);
  return at::AtenIpexTypeXPU::softplus_tanh_mul(_self, beta, threshold, _mul);
}

at::Tensor permute_contiguous(
    const at::Tensor& self,
    at::IntArrayRef dims,
    at::MemoryFormat dim_contiguous) {
  Tensor result;
  // plain format tensor will go through naitve permute contiguous pass
  if (DPCPPTensorContext::get_tensor_ctx(self).is_plain()) {
    result = at::native::permute(self, dims).contiguous(dim_contiguous);
    return result;
  }
  // block format tensor will be reordered to plain format in this fusion, and
  // it mainly consists of 4 steps.

  // 1. run some checks and calculate the output tensor shape.
  auto nDims = self.dim();
  TORCH_CHECK(
      dims.size() == (size_t)nDims, "number of dims don't match in permute");
  auto oldSizes = self.sizes();
  auto oldStrides = self.strides();
  DimVector newSizes(nDims);
  DimVector newStrides(nDims);
  std::vector<bool> seen(nDims);
  for (const auto i : c10::irange(nDims)) {
    auto dim = at::maybe_wrap_dim(dims[i], nDims);
    TORCH_CHECK(!seen[dim], "repeated dim in permute");
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    newStrides[i] = oldStrides[dim];
  }
  DimVector revert_dim(nDims);

  // 2.calculate reverse permute index for new tensor.
  for (const auto i : c10::irange(nDims)) {
    revert_dim[dims[i]] = i;
  }
  if (self.is_quantized()) {
    result = at::_empty_affine_quantized(
        newSizes,
        self.options(),
        self.q_scale(),
        self.q_zero_point(),
        dim_contiguous);

  } else {
    result = at::empty(newSizes, self.options(), dim_contiguous);
  }

  // 3.permute the new contiguous tensor to same shape against input.
  Tensor permute_one = at::native::permute(result, revert_dim);

  // 4.reorder the input tensor to plain format and put it into the new tensor,
  // which will be contiguous in the shape of the desire output one.
  ::xpu::oneDNN::reorder(self, permute_one);
  result = at::native::permute(permute_one, dims);
  return result;
}

at::Tensor q_conv2d_dequantize_softplus_tanh_mul(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    const Scalar& beta,
    const Scalar& threshold) {
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::q_conv2d_dequantize_softplus_tanh_mul(
      input, packed_weight, output_scale, output_zero_point, beta, threshold);
}

at::Tensor q_conv2d_dequantize_softplus_tanh_mul_quantize(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    const Scalar& beta,
    const Scalar& threshold,
    double q_scale,
    int64_t q_zpoint,
    at::ScalarType dtype) {
  RECORD_FUNCTION(
      "q_conv2d_dequantize_softplus_tanh_mul_quantize",
      std::vector<c10::IValue>({input}));
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::q_conv2d_dequantize_softplus_tanh_mul_quantize(
      input,
      packed_weight,
      output_scale,
      output_zero_point,
      beta,
      threshold,
      q_scale,
      q_zpoint,
      dtype);
}

at::Tensor q_conv2d_dequantize_softplus_tanh_mul_quantize_add(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    const Scalar& beta,
    const Scalar& threshold,
    double q_scale,
    int64_t q_zpoint,
    at::ScalarType dtype,
    Tensor qb,
    double add_scale,
    int64_t add_zero_point) {
  RECORD_FUNCTION(
      "q_conv2d_dequantize_softplus_tanh_mul_quantize_add",
      std::vector<c10::IValue>({input}));
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::
      q_conv2d_dequantize_softplus_tanh_mul_quantize_add(
          input,
          packed_weight,
          output_scale,
          output_zero_point,
          beta,
          threshold,
          q_scale,
          q_zpoint,
          dtype,
          qb,
          add_scale,
          add_zero_point);
}

at::Tensor conv2d_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  RECORD_FUNCTION(
      "conv2d_relu", std::vector<c10::IValue>({input, weight, bias}));
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::convolution_relu(
      input, weight, bias, stride, padding, dilation, false, {{0, 0}}, groups);
}

at::Tensor pad_conv2d(
    const at::Tensor& input,
    at::IntArrayRef pad_nd,
    Scalar value,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  RECORD_FUNCTION(
      "pad_conv2d", std::vector<c10::IValue>({input, weight, bias}));
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::pad_convolution(
      input,
      pad_nd,
      value,
      weight,
      bias,
      stride,
      padding,
      dilation,
      false,
      {{0, 0}},
      groups);
}

at::Tensor conv2d_sigmoid(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  RECORD_FUNCTION(
      "conv2d_sigmoid", std::vector<c10::IValue>({input, weight, bias}));
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::convolution_sigmoid(
      input, weight, bias, stride, padding, dilation, false, {{0, 0}}, groups);
}

at::Tensor mul_add(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Tensor& accumu,
    at::Scalar alpha) {
  RECORD_FUNCTION("mul_add", std::vector<c10::IValue>({self, other, accumu}));
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::AtenIpexTypeXPU::mul_add(self, other, accumu, alpha);
}

at::Tensor dequant_pixelshuffle(
    const at::Tensor& self,
    int64_t upscale_factor) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::empty_like(self);
}

at::Tensor dequant_pixelshuffle_quant(
    const at::Tensor& self,
    int64_t upscale_factor,
    double scale,
    int64_t zero_pad,
    at::ScalarType dtype) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::pixel_shuffle(self, upscale_factor);
}

at::Tensor q_conv2d_sum_relu(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double conv_scale,
    int64_t conv_zpoint,
    at::Tensor& accumu,
    double sum_scale,
    int64_t sum_zpoint) {
  RECORD_FUNCTION(
      "q_conv2d_sum_relu", std::vector<c10::IValue>({input, packed_weight}));
  const OptionalDeviceGuard device_guard(device_of(accumu));
  return at::AtenIpexTypeXPU::q_conv2d_sum_relu(
      accumu,
      input,
      packed_weight,
      conv_scale,
      conv_zpoint,
      sum_scale,
      sum_zpoint);
}

at::Tensor q_conv2d_sigmoid(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zpoint) {
  RECORD_FUNCTION(
      "q_conv2d_sigmoid", std::vector<c10::IValue>({input, packed_weight}));
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::q_conv2d_sigmoid(
      input, packed_weight, output_scale, output_zpoint);
}

at::Tensor q_conv2d_leaky_relu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zpoint,
    Scalar negative_slope) {
  RECORD_FUNCTION(
      "q_conv2d_leaky_relu", std::vector<c10::IValue>({input, packed_weight}));
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::q_conv2d_leaky_relu(
      input, packed_weight, output_scale, output_zpoint, negative_slope);
}

at::Tensor batch_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double momentum,
    double eps,
    bool use_dnn) {
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::empty_like(input);
}

at::Tensor fold_weight(
    const at::Tensor& weight,
    const at::Tensor& bn_weight,
    const at::Tensor& running_var,
    float eps) {
  const OptionalDeviceGuard device_guard(device_of(weight));
  return at::empty_like(weight);
}

at::Tensor fold_bias(
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& bn_weight,
    const at::Tensor& bn_bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    float eps) {
  const OptionalDeviceGuard device_guard(device_of(weight));
  return at::empty_like(bias);
}

at::Tensor reorder(
    const at::Tensor& input,
    dnnl::memory::format_tag from,
    dnnl::memory::format_tag to,
    int64_t groups) {
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::empty_like(input);
}

at::Tensor matmul_add(
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Tensor& accumul1,
    Scalar beta1) {
  const OptionalDeviceGuard device_guard(device_of(accumul1));
  return at::AtenIpexTypeXPU::matmul_add(tensor1, tensor2, accumul1, beta1);
}

at::Tensor trans_matmul(
    const at::Tensor& tensor2,
    int dim1,
    int dim2,
    const at::Tensor& tensor1) {
  // const OptionalDeviceGuard device_guard(device_of(accumul1));
  return at::AtenIpexTypeXPU::trans_matmul(tensor2, dim1, dim2, tensor1);
}

at::Tensor t_matmul(const at::Tensor& tensor2, const at::Tensor& tensor1) {
  // const OptionalDeviceGuard device_guard(device_of(accumul1));
  return at::AtenIpexTypeXPU::t_matmul(tensor2, tensor1);
}

at::Tensor t_matmul_add(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1,
    at::Tensor& accumul1,
    Scalar beta1) {
  const OptionalDeviceGuard device_guard(device_of(accumul1));
  return at::AtenIpexTypeXPU::t_matmul_add(tensor2, tensor1, accumul1, beta1);
}

at::Tensor t_matmul_add_gelu(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1,
    at::Tensor& accumul1,
    Scalar beta1) {
  const OptionalDeviceGuard device_guard(device_of(accumul1));
  return at::AtenIpexTypeXPU::t_matmul_add_gelu(
      tensor2, tensor1, accumul1, beta1);
}

at::Tensor t_matmul_add_add(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1,
    at::Tensor& accumul1,
    Scalar beta1,
    at::Tensor& accumul2,
    Scalar beta2) {
  return at::AtenIpexTypeXPU::t_matmul_add_add(
      tensor2, tensor1, accumul1, beta1, accumul2, beta2);
}

at::Tensor trans_matmul_div(
    const at::Tensor& tensor2,
    int dim1,
    int dim2,
    const at::Tensor& tensor1,
    at::Scalar oscale) {
  // const OptionalDeviceGuard device_guard(device_of(accumul1));
  return at::AtenIpexTypeXPU::trans_matmul_div(
      tensor2, dim1, dim2, tensor1, oscale);
}

at::Tensor linear_gelu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::linear_gelu(input, weight, bias);
}

at::Tensor linear_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::linear_relu(input, weight, bias);
}

at::Tensor linear_sigmoid(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::linear_sigmoid(input, weight, bias);
}

at::Tensor linear_add(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& accumu,
    at::Scalar alpha) {
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::linear_add(input, weight, bias, accumu, alpha);
}

at::Tensor _convolution_relu(
    at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride_,
    at::IntArrayRef padding_,
    at::IntArrayRef dilation_,
    bool transposed_,
    at::IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {
  RECORD_FUNCTION(
      "_convolution_relu", std::vector<c10::IValue>({input, weight, bias}));
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::convolution_relu(
      input,
      weight,
      bias,
      stride_,
      padding_,
      dilation_,
      transposed_,
      output_padding_,
      groups_,
      1.0,
      0.0,
      0.0);
}

at::Tensor _convolution_sum(
    at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride_,
    at::IntArrayRef padding_,
    at::IntArrayRef dilation_,
    bool transposed_,
    at::IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    at::Tensor& accum,
    at::Scalar scale) {
  RECORD_FUNCTION(
      "_convolution_sum", std::vector<c10::IValue>({input, weight, bias}));
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::convolution_sum(
      input,
      weight,
      bias,
      stride_,
      padding_,
      dilation_,
      transposed_,
      output_padding_,
      groups_,
      accum,
      scale,
      0.0,
      0.0);
}

at::Tensor _convolution_sum_relu(
    at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride_,
    at::IntArrayRef padding_,
    at::IntArrayRef dilation_,
    bool transposed_,
    at::IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    at::Tensor& accum,
    at::Scalar scale) {
  RECORD_FUNCTION(
      "_convolution_sum_add", std::vector<c10::IValue>({input, weight, bias}));
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::convolution_sum_relu(
      input,
      weight,
      bias,
      stride_,
      padding_,
      dilation_,
      transposed_,
      output_padding_,
      groups_,
      accum,
      scale,
      0.0,
      0.0);
}

at::Tensor _convolution_silu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride_,
    at::IntArrayRef padding_,
    at::IntArrayRef dilation_,
    bool transposed_,
    at::IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {
  RECORD_FUNCTION(
      "_convolution_silu", std::vector<c10::IValue>({input, weight, bias}));
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::AtenIpexTypeXPU::convolution_silu(
      input,
      weight,
      bias,
      stride_,
      padding_,
      dilation_,
      transposed_,
      output_padding_,
      groups_,
      1.0,
      1.0,
      0.0);
}

} // namespace xpu
} // namespace jit
} // namespace torch
