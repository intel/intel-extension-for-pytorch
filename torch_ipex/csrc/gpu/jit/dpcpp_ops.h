#pragma once
#include <ATen/ATen.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>

#include <oneapi/dnnl/dnnl.hpp>

namespace torch {
namespace jit {
namespace dpcpp {

at::Tensor& conv2d_sum(at::Tensor& accumu,
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, at::Scalar alpha);

at::Tensor& conv2d_sum_relu(at::Tensor& accumu,
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, at::Scalar alpha);

at::Tensor conv2d_relu(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups);

at::Tensor conv2d_sigmoid(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups);

at::Tensor matmul_sum(at::Tensor& accumu,
    const at::Tensor& m1, const at::Tensor& m2, at::Scalar alpha);

at::Tensor trans_matmul_scale_sum(at::Tensor& accumu, const at::Tensor& tensor1,
    const at::Tensor& tensor2, at::Scalar oscale, at::Scalar alpha);

at::Tensor mul_add(const at::Tensor& self,
    const at::Tensor& other, const at::Tensor& accumu, at::Scalar alpha);

at::Tensor batch_norm(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& running_mean,
    const at::Tensor& running_var, bool train, double momentum, double eps, bool use_dnn);

at::Tensor fold_weight(
    const at::Tensor& weight, const at::Tensor& bn_weight, const at::Tensor& running_var, float eps);

at::Tensor fold_bias(
    const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& bn_weight,
    const at::Tensor& bn_bias, const at::Tensor& running_mean, const at::Tensor& running_var, float eps);

at::Tensor reorder(
    const at::Tensor& input,
    dnnl::memory::format_tag from, dnnl::memory::format_tag to, int64_t groups);

at::Tensor q_conv2d_sum_relu(
    at::Tensor& accumu, const at::Tensor& input, const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
    double conv_scale, int64_t conv_zpoint, double sum_scale,
    int64_t sum_zpoint);

} // dpcpp
} // jit
} // torch
