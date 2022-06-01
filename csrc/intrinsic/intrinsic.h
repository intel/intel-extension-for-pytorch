#pragma once

#include <ATen/ATen.h>
#include <ATen/native/quantized/packed_params.h>
#include <tensor/Context.h>

namespace at {
namespace AtenIpexTypeXPU {

struct DPCPPTensorContext;

at::Tensor interaction(at::Tensor& input_mlp, at::Tensor& input_emb);

void fused_ADAMW(
    at::Tensor& param_,
    at::Tensor& exp_avg_,
    at::Tensor& exp_avg_sq_,
    at::Tensor& max_exp_avg_sq_,
    at::Tensor& grad_,
    at::Tensor& param2_,
    const bool amsgrad,
    const double step,
    const double beta1,
    const double beta2,
    const double learning_rate,
    const double weight_decay,
    const double eps);

c10::optional<at::Tensor> fused_SGD(
    at::Tensor& fp32_weight,
    at::Tensor& grad,
    const c10::optional<at::Tensor>& momentum_buffer_,
    at::Tensor& weight,
    const double momentum,
    const double lr,
    const double weight_decay,
    const double dampening,
    const bool nesterov);

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

at::Tensor _convolution_relu_(
    at::Tensor& input,
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

at::Tensor convolution_silu(
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
    at::Scalar alpha = 1.f,
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

at::Tensor pad_convolution(
    const at::Tensor& input,
    at::IntArrayRef pad_nd,
    Scalar value,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups);

at::Tensor& fill_slice_with_index(at::Tensor& t, int dim);

at::Tensor& std_var_out(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef dim,
    int64_t correction_opt,
    bool keepdim,
    bool take_sqrt);

std::tuple<Tensor&, Tensor&> std_var_mean_out(
    const char* fname,
    Tensor& result1,
    Tensor& result2,
    const Tensor& self,
    IntArrayRef dim,
    int64_t correction_opt,
    bool keepdim,
    bool take_sqrt);

at::Tensor matmul_add(
    const at::Tensor& m1,
    const at::Tensor& m2,
    at::Tensor& accumu,
    Scalar beta);

at::Tensor trans_matmul(
    const at::Tensor& tensor2,
    int dim1,
    int dim2,
    const at::Tensor& tensor1);

at::Tensor t_matmul(const at::Tensor& tensor2, const at::Tensor& tensor1);

at::Tensor t_matmul_add(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1,
    at::Tensor& accumul1,
    Scalar beta1);

at::Tensor t_matmul_add_gelu(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1,
    at::Tensor& accumul1,
    Scalar beta1);

at::Tensor t_matmul_add_add(
    const at::Tensor& tensor2,
    const at::Tensor& tensor1,
    at::Tensor& accumul1,
    Scalar beta1,
    at::Tensor& accumul2,
    Scalar beta2);

at::Tensor trans_matmul_div(
    const at::Tensor& tensor2,
    int dim1,
    int dim2,
    const at::Tensor& tensor1,
    Scalar oscale);

at::Tensor linear_gelu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias);

at::Tensor linear_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias);

at::Tensor linear_sigmoid(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias);

at::Tensor linear_add(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& accumu,
    at::Scalar alpha);

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

at::Tensor q_conv2d_leaky_relu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Scalar negative_slope);

at::Tensor q_conv2d_sigmoid(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point);

at::Tensor q_conv2d_dequantize(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point);

at::Tensor softplus_tanh(
    const Tensor& self,
    const Scalar& beta,
    const Scalar& threshold);

at::Tensor softplus_tanh_mul(
    const Tensor& self,
    const Scalar& beta,
    const Scalar& threshold,
    const Tensor& mul_input);

at::Tensor q_conv2d_dequantize_softplus_tanh_mul(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    const Scalar& beta,
    const Scalar& threshold);

at::Tensor q_conv2d_dequantize_softplus_tanh_mul_quantize(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    const Scalar& beta,
    const Scalar& threshold,
    double q_scale,
    int64_t q_zpoint,
    at::ScalarType dtype);

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
    int64_t add_zero_point);

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
