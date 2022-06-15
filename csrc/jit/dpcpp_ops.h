#pragma once
#include <ATen/ATen.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <oneDNN/oneDNN.h>
#include <oneapi/dnnl/dnnl.hpp>

namespace torch {
namespace jit {
namespace xpu {

at::Tensor matmul_fusion_variants(
    at::Tensor& accumul1,
    at::Tensor& accumul2,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    float oscale,
    float beta1,
    float beta2,
    bool trans,
    int fusion_type = 0);

at::Tensor matmul_fusion_variants_gelu(
    at::Tensor& accumul1,
    at::Tensor& accumul2,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    float oscale,
    float beta1,
    float beta2,
    bool trans);

at::Tensor matmul_fusion_variants_dropout(
    at::Tensor& accumul1,
    at::Tensor& accumul2,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    float oscale,
    float beta1,
    float beta2,
    bool trans,
    double p,
    bool train,
    bool inplace);

at::Tensor& conv2d_sum(
    at::Tensor& accumu,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    at::Scalar alpha);

at::Tensor& conv2d_sum_relu(
    at::Tensor& accumu,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    at::Scalar alpha);

at::Tensor conv2d_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups);

at::Tensor conv2d_sigmoid(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups);

at::Tensor mul_add(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Tensor& accumu,
    at::Scalar alpha);

at::Tensor dequant_pixelshuffle(const at::Tensor& self, int64_t upscale_factor);

at::Tensor dequant_pixelshuffle_quant(
    const at::Tensor& self,
    int64_t upscale_factor,
    double scale,
    int64_t zero_pad,
    at::ScalarType dtype);

at::Tensor batch_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double momentum,
    double eps,
    bool use_dnn);

at::Tensor fold_weight(
    const at::Tensor& weight,
    const at::Tensor& bn_weight,
    const at::Tensor& running_var,
    float eps);

at::Tensor fold_bias(
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& bn_weight,
    const at::Tensor& bn_bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    float eps);

at::Tensor reorder(
    const at::Tensor& input,
    dnnl::memory::format_tag from,
    dnnl::memory::format_tag to,
    int64_t groups);

at::Tensor q_conv2d_sum_relu(
    at::Tensor& accumu,
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double conv_scale,
    int64_t conv_zpoint,
    double sum_scale,
    int64_t sum_zpoint);

at::Tensor q_conv2d_leaky_relu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double out_scale,
    int64_t out_zpoint,
    Scalar negative_scope);

at::Tensor q_conv2d_sigmoid(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zpoint);

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

at::Tensor trans_addmm(
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& input,
    at::Scalar beta,
    at::Scalar alpha);

at::Tensor trans_addmm_relu(
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& input,
    at::Scalar beta,
    at::Scalar alpha);

at::Tensor trans_addmm_sigmoid(
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& input,
    at::Scalar beta,
    at::Scalar alpha);

at::Tensor trans_addmm_dropout(
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& input,
    at::Scalar beta,
    at::Scalar alpha,
    double p,
    bool train,
    bool inplace);

} // namespace xpu
} // namespace jit
} // namespace torch
