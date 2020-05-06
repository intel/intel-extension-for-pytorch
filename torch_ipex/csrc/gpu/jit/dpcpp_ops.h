#pragma once
#include <ATen/ATen.h>
#include <dnnl.hpp>


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

at::Tensor batch_norm(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& running_mean,
    const at::Tensor& running_var, bool train, double momentum, double eps, bool use_cuda);

at::Tensor fold_weight(
    const at::Tensor& weight, const at::Tensor& bn_weight, const at::Tensor& running_var, float eps);

at::Tensor fold_bias(
    const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& bn_weight,
    const at::Tensor& bn_bias, const at::Tensor& running_mean, const at::Tensor& running_var, float eps);

at::Tensor reorder(
    const at::Tensor& input,
    dnnl::memory::format_tag from, dnnl::memory::format_tag to, int64_t groups);

} // dpcpp
} // jit
} // torch
