#pragma once
#include <ATen/ATen.h>
#include <ATen/native/quantized/PackedParams.h>
#include <oneDNN/oneDNN.h>
#include <oneapi/dnnl/dnnl.hpp>
#include "torch/library.h"

namespace torch {
namespace jit {
namespace xpu {

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
    double eps);

at::Tensor fold_bias(
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& bn_weight,
    const at::Tensor& bn_bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    double eps);

at::Tensor reorder(
    const at::Tensor& input,
    dnnl::memory::format_tag from,
    dnnl::memory::format_tag to,
    int64_t groups);

} // namespace xpu
} // namespace jit
} // namespace torch
