#pragma once

#include <ideep.hpp>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at { namespace native {

at::Tensor dnnl_conv2d(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);

at::Tensor dnnl_conv2d_relu(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);

at::Tensor& dnnl_conv2d_sum(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, at::Tensor& accumu, at::Scalar alpha);

at::Tensor& dnnl_conv2d_sum_relu(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups, at::Tensor& accumu, at::Scalar alpha);

at::Tensor dnnl_reorder(
    const at::Tensor& input, ideep::format from, ideep::format to, int64_t groups = 1);

at::Tensor dnnl_relu(const at::Tensor& input);

at::Tensor& dnnl_relu_(at::Tensor& input);

at::Tensor dnnl_sum(
    const at::Tensor& self, const at::Tensor& other, at::Scalar alpha);

at::Tensor& dnnl_sum_(
    at::Tensor& self, const at::Tensor& other, at::Scalar alpha);

at::Tensor dnnl_pooling_max_2d(
  const at::Tensor& self,
  at::IntArrayRef kernel_size,
  at::IntArrayRef stride,
  at::IntArrayRef padding,
  at::IntArrayRef dilation,
  bool ceil_mode);

at::Tensor dnnl_pooling_avg_2d(
  const at::Tensor& self,
  at::IntArrayRef kernel_size,
  at::IntArrayRef stride,
  at::IntArrayRef padding,
  bool ceil_mode);

at::Tensor dnnl_batch_norm(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& running_mean,
    const at::Tensor& running_var, bool train, double momentum, double eps, bool use_cuda);

at::Tensor dnnl_fold_weight(const at::Tensor& weight, const at::Tensor& bn_weight, const at::Tensor& running_var, float eps);

at::Tensor dnnl_fold_bias(
    const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& bn_weight,
    const at::Tensor& bn_bias, const at::Tensor& running_mean, const at::Tensor& running_var, float eps);

}}
