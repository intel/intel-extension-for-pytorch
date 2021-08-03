#pragma once

#include <ATen/ATen.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <tensor/Context.h>

namespace at {
namespace AtenIpexTypeXPU {

struct DPCPPTensorContext;

at::Tensor& fused_adamW(
    at::Tensor& grad_input,
    const at::Tensor& avg,
    const at::Tensor& avg_sq,
    int64_t step = 1.0,
    double lr = 1.0,
    double eps = 1.0,
    double beta1 = 1.0,
    double beta2 = 1.0,
    double weight_decay = 0.f,
    const bool correct_bias = true);

at::Tensor convolution_sum(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::Tensor& accumu,
    at::Scalar scale = 1.0,
    at::Scalar alpha = 0.f,
    at::Scalar beta = 0.f);

at::Tensor convolution_sum_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::Tensor& accumu,
    at::Scalar scale = 1.0,
    at::Scalar alpha = 0.f,
    at::Scalar beta = 0.f);

at::Tensor convolution_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::Scalar scale = 1.0,
    at::Scalar alpha = 0.f,
    at::Scalar beta = 0.f);

at::Tensor convolution_sigmoid(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::Scalar scale = 1.0,
    at::Scalar alpha = 0.f,
    at::Scalar beta = 0.f);

at::Tensor matmul_sum(
    at::Tensor& accumu,
    const at::Tensor& m1,
    const at::Tensor& m2,
    at::Scalar beta);

at::Tensor& trans_baddbmm_out(
    at::Tensor& result,
    const at::Tensor& input,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    Scalar beta,
    Scalar alpha);

at::Tensor& fill_slice_with_index(at::Tensor& t, int dim);

at::Tensor& std_var_out(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim,
    bool take_sqrt);

std::tuple<Tensor&, Tensor&> std_var_mean_out(
    const char* fname,
    Tensor& result1,
    Tensor& result2,
    const Tensor& self,
    IntArrayRef dim,
    bool unbiased,
    bool keepdim,
    bool take_sqrt);

at::Tensor linear_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Scalar beta = 1.0f,
    at::Scalar alpha = 1.0f);

at::Tensor linear_sigmoid(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Scalar beta = 1.0f,
    at::Scalar alpha = 1.0f);

at::Tensor trans_linear(
    const at::Tensor& input,
    const at::Tensor& m1,
    const at::Tensor& m2,
    at::Scalar beta = 1.0f,
    at::Scalar alpha = 1.0f);

at::Tensor mul_add(
    const Tensor& self,
    const Tensor& other,
    const Tensor& accumu,
    Scalar alpha);

at::Tensor packed_add(
    at::Tensor& top_half,
    at::Tensor& bot_half,
    const at::Tensor& grad,
    float alpha);

at::Tensor fusion_amdd(
    at::Tensor& p,
    at::Tensor& d_p,
    at::Tensor& buf,
    float weight_decay,
    float momentum,
    float dampening,
    float lr);

at::Tensor empty_opaque_tensor(
    DPCPPTensorContext::Meta meta,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format);

at::Tensor empty_opaque_qtensor(
    DPCPPTensorContext::Meta meta,
    c10::optional<MemoryFormat> optional_memory_format,
    QuantizerPtr quantizer);

at::Tensor to_plain_if_needed(const Tensor& tensor);

at::Tensor to_plain_if_needed_(const Tensor& tensor);

std::vector<at::Tensor> to_plain_if_needed(TensorList tensor);

at::Tensor new_qtensor(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer);

at::Tensor dequant_pixelshuffle(const Tensor& self, int64_t upscale_factor);

at::Tensor dequant_pixelshuffle_quant(
    const Tensor& self,
    int64_t upscale_factor,
    double scale,
    int64_t zero_pad,
    at::ScalarType dtype);

at::Tensor q_conv2d_sum_relu(
    at::Tensor& accumu,
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double conv_scale,
    int64_t conv_zero_point,
    double sum_scale,
    int64_t sum_zero_point);

at::Tensor quantize_tensor_per_tensor_affine(
    at::Tensor& qtensor,
    const at::Tensor& rtensor,
    double scale,
    int64_t zero_point);

at::Tensor quantize_tensor_per_channel_affine(
    at::Tensor& qtensor,
    const at::Tensor& rtensor,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    int64_t axis);

at::Tensor dequantize_tensor_per_tensor_affine(
    at::Tensor& rtensor,
    const at::Tensor& qtensor,
    double scale,
    int64_t zero_point);

at::Tensor dequantize_tensor_per_channel_affine(
    at::Tensor& rtensor,
    const at::Tensor& qtensor,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    int64_t axis);

} // namespace AtenIpexTypeXPU
} // namespace at

namespace at {
namespace AtenIpexTypeQuantizedXPU {

at::Tensor trans_linear(
    const at::Tensor& input,
    const at::Tensor& m1,
    const at::Tensor& m2,
    at::Scalar beta = 1.0f,
    at::Scalar alpha = 1.0f);

}
} // namespace at
