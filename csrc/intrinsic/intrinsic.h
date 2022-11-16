#pragma once

#include <ATen/ATen.h>
#include <ATen/native/quantized/PackedParams.h>
#include <utils/Macros.h>

#include <aten/operators/Linear.h>
#include "c10/util/string_view.h"

namespace at {
namespace AtenIpexTypeXPU {

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking);

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

at::Tensor pad_convolution(
    const at::Tensor& input,
    at::IntArrayRef pad_nd,
    Scalar value,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups);

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
    Scalar beta1,
    c10::string_view approximate);

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

at::Tensor linear_silu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias);

at::Tensor linear_gelu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    c10::string_view approximate);

at::Tensor linear_sum(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& accumu,
    at::Scalar alpha);

at::Tensor mul_add(
    const Tensor& self,
    const Tensor& other,
    const Tensor& accumu,
    Scalar alpha);

at::Tensor mul_add_scalar(
    const Tensor& self,
    Scalar other,
    const Tensor& accumu,
    Scalar alpha);

at::Tensor packed_add(
    at::Tensor& top_half,
    at::Tensor& bot_half,
    const at::Tensor& grad,
    float alpha);

at::Tensor to_plain_if_needed(const Tensor& tensor);

at::Tensor dequant_pixelshuffle(const Tensor& self, int64_t upscale_factor);

at::Tensor dequant_pixelshuffle_quant(
    const Tensor& self,
    int64_t upscale_factor,
    double scale,
    int64_t zero_pad,
    at::ScalarType dtype);

at::Tensor q_conv2d_dequantize(
    const at::Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point);

at::Tensor q_conv2d_sigmoid(
    const Tensor& input,
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

at::Tensor permute_contiguous(
    const at::Tensor& self,
    at::IntArrayRef dims,
    at::MemoryFormat dim_contiguous);

/* DECALRE_CONV
This macro is used to convinentlt generate conv related post op declaration when
no extra parameters are brought in.
*/
#define DECLARE_CONV(op, ...)                                           \
  at::Tensor q_conv2d_##op(                                             \
      const Tensor& input,                                              \
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight, \
      double output_scale,                                              \
      int64_t output_zero_point);                                       \
                                                                        \
  at::Tensor convolution_##op(                                          \
      const at::Tensor& input,                                          \
      const at::Tensor& weight,                                         \
      const c10::optional<at::Tensor>& bias,                            \
      std::vector<int64_t> stride,                                      \
      std::vector<int64_t> padding,                                     \
      std::vector<int64_t> dilation,                                    \
      int64_t groups);                                                  \
                                                                        \
  Tensor _convolution_##op(                                             \
      const Tensor& input,                                              \
      const Tensor& weight,                                             \
      const c10::optional<at::Tensor>& bias,                            \
      std::vector<int64_t> stride_,                                     \
      std::vector<int64_t> padding_,                                    \
      std::vector<int64_t> dilation_,                                   \
      bool transposed,                                                  \
      std::vector<int64_t> output_padding_,                             \
      int groups,                                                       \
      bool benchmark,                                                   \
      bool deterministic,                                               \
      bool cudnn_enabled,                                               \
      bool allow_tf32);

#define DECLARE_LINEAR(op) \
  at::Tensor linear_##op(  \
      const Tensor& input, const Tensor& weight, const Tensor& bias);

DECLARE_CONV(sqrt)
DECLARE_CONV(abs)
DECLARE_CONV(tanh)
DECLARE_CONV(square)
DECLARE_CONV(exp)
DECLARE_CONV(log)
DECLARE_CONV(round)
DECLARE_CONV(log_sigmoid)
DECLARE_CONV(hardswish)
DECLARE_CONV(mish)
DECLARE_CONV(silu)
DECLARE_CONV(hardsigmoid)
DECLARE_CONV(sigmoid)
DECLARE_CONV(relu)

Tensor convolution_gelu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    int64_t groups_,
    c10::string_view approximate);

Tensor _convolution_gelu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    c10::string_view approximate);

at::Tensor q_conv2d_gelu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    c10::string_view approximate);

at::Tensor q_conv2d_pow(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Scalar pow);

at::Tensor convolution_pow(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups,
    Scalar pow);

Tensor _convolution_pow(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Scalar pow);

at::Tensor q_conv2d_leaky_relu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Scalar negative_slope);

at::Tensor convolution_leaky_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups,
    Scalar negative_slope);

Tensor _convolution_leaky_relu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Scalar negative_slope);

at::Tensor q_conv2d_hardtanh(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Scalar minval,
    Scalar maxval);

at::Tensor convolution_hardtanh(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups,
    Scalar minval,
    Scalar maxval);

Tensor _convolution_hardtanh(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Scalar minval,
    Scalar maxval);

at::Tensor q_conv2d_elu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale);

at::Tensor convolution_elu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale);

Tensor _convolution_elu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    std::vector<int64_t> stride_,
    std::vector<int64_t> padding_,
    std::vector<int64_t> dilation_,
    bool transposed,
    std::vector<int64_t> output_padding_,
    int groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale);

Tensor convolution_sum(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_,
    Tensor& accumu,
    Scalar scale);

Tensor _convolution_sum(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Tensor& accumu,
    Scalar scale);

at::Tensor q_conv2d_sum(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Tensor& accumu,
    float sum_scale,
    int sum_zero_point);

Tensor convolution_sum_relu(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_,
    Tensor& accumu,
    Scalar scale);

Tensor _convolution_sum_relu(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32,
    Tensor& accumu,
    Scalar scale);

at::Tensor q_conv2d_sum_relu(
    const Tensor& input,
    const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double output_scale,
    int64_t output_zero_point,
    Tensor& accumu,
    float sum_scale,
    int sum_zero_point);

at::Tensor convolution_binary_mul(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_,
    const Tensor& binary);

DECLARE_LINEAR(sqrt)
DECLARE_LINEAR(abs)
DECLARE_LINEAR(tanh)
DECLARE_LINEAR(square)
DECLARE_LINEAR(exp)
DECLARE_LINEAR(log)
DECLARE_LINEAR(round)
DECLARE_LINEAR(log_sigmoid)
DECLARE_LINEAR(hardswish)
DECLARE_LINEAR(mish)
DECLARE_LINEAR(hardsigmoid)
DECLARE_LINEAR(sigmoid)
DECLARE_LINEAR(relu)

} // namespace AtenIpexTypeXPU
} // namespace at

namespace xpu {
namespace dpcpp {

/*
 * The namespace at::AtenIpexTypeXPU only serves as operator/kernel
 * implementation. We export operators here under xpu::dpcpp namespace for
 * frontend usage.
 */
EXPORT_TO_XPU(interaction);
EXPORT_TO_XPU(fused_ADAMW);
EXPORT_TO_XPU(fused_SGD);
EXPORT_TO_XPU(to_plain_if_needed);
EXPORT_TO_XPU(packed_add);
EXPORT_TO_XPU(mul_add);
EXPORT_TO_XPU(linear_gelu);
EXPORT_TO_XPU(linear_relu);
EXPORT_TO_XPU(linear_sigmoid);

EXPORT_TO_XPU_ALIAS(copy_, direct_copy);

} // namespace dpcpp
} // namespace xpu
