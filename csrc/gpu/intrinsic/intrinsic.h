#pragma once

#include <ATen/ATen.h>
#include <ATen/native/quantized/PackedParams.h>
#include <utils/Macros.h>

#include <aten/operators/Linear.h>
#include <c10/util/string_view.h>

namespace at {
namespace AtenIpexTypeXPU {

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking);

at::Tensor interaction(at::Tensor& input_mlp, at::Tensor& input_emb);

void adamw_fused_step(
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

c10::optional<at::Tensor> sgd_fused_step(
    at::Tensor& fp32_weight,
    at::Tensor& grad,
    const c10::optional<at::Tensor>& momentum_buffer_,
    at::Tensor& weight,
    const double momentum,
    const double lr,
    const double weight_decay,
    const double dampening,
    const bool nesterov);

at::Tensor packed_add(
    at::Tensor& top_half,
    at::Tensor& bot_half,
    const at::Tensor& grad,
    double alpha);

at::Tensor to_plain_if_needed(const Tensor& tensor);

at::Tensor dequant_pixelshuffle(const Tensor& self, int64_t upscale_factor);

at::Tensor dequant_pixelshuffle_quant(
    const Tensor& self,
    int64_t upscale_factor,
    double scale,
    int64_t zero_pad,
    at::ScalarType dtype);

} // namespace AtenIpexTypeXPU
} // namespace at

namespace xpu {
namespace dpcpp {

/*
 * The namespace at::AtenIpexTypeXPU only serves as operator/kernel
 * implementation. We export operators here under xpu::dpcpp namespace for
 * frontend usage.
 */
EXPORT_TO_XPU(to_plain_if_needed);
EXPORT_TO_XPU_ALIAS(copy_, direct_copy);

} // namespace dpcpp
} // namespace xpu
