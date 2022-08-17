#pragma once

#include <ATen/ATen.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <tensor/Context.h>

namespace at {
namespace AtenIpexTypeXPU {

struct DPCPPTensorContext;

void matmul(
    Tensor& result,
    const Tensor& m1,
    const Tensor& m2,
    const Tensor& b,
    const Tensor& po,
    float beta,
    float alpha,
    bool m2_trans,
    int fusion);

at::Tensor interaction(at::Tensor& input_mlp, at::Tensor& input_emb);

at::Tensor& fused_adamWMasterWeight(
    at::Tensor& master_weight,
    at::Tensor& weight,
    at::Tensor& grad,
    const bool amsgrad,
    at::Tensor& avg,
    at::Tensor& avg_sq,
    at::Tensor& max_avg_sq,
    int64_t& step,
    double lr,
    double eps,
    double beta1,
    double beta2,
    double weight_decay);

at::Tensor& transformer_adamWMasterWeight(
    at::Tensor& master_weight,
    at::Tensor& weight,
    at::Tensor& grad,
    at::Tensor& avg,
    at::Tensor& avg_sq,
    at::Tensor& max_avg_sq,
    int64_t& step,
    double lr,
    double eps,
    double beta1,
    double beta2,
    double weight_decay,
    const bool correct_bias);

at::Tensor& fused_SGDMasterWeight(
    at::Tensor& master_weight,
    at::Tensor& weight,
    at::Tensor& grad,
    double weight_decay,
    bool momentum_buffer_existed,
    at::Tensor& momentum_buffer,
    double momentum,
    double dampening,
    bool nesterov,
    double lr);

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

at::Tensor matmul_add(
    at::Tensor& accumu,
    const at::Tensor& m1,
    const at::Tensor& m2,
    at::Scalar beta);

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

at::Tensor trans_addmm_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Scalar beta = 1.0f,
    at::Scalar alpha = 1.0f);

at::Tensor trans_addmm_sigmoid(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Scalar beta = 1.0f,
    at::Scalar alpha = 1.0f);

at::Tensor trans_addmm(
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

namespace at {
namespace AtenIpexTypeQuantizedXPU {

at::Tensor trans_addmm(
    const at::Tensor& input,
    const at::Tensor& m1,
    const at::Tensor& m2,
    at::Scalar beta = 1.0f,
    at::Scalar alpha = 1.0f);
}
} // namespace at
