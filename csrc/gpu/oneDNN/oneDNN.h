#pragma once

#include <ATen/ATen.h>
#include <oneDNN/Runtime.h>
#include <oneDNN/Utils.h>
#include <utils/LRUCache.h>

#include "BatchNorm.h"
#include "Binary.h"
#include "Concat.h"
#include "ConvUtils.h"
#include "Deconv.h"
#include "Eltwise.h"
#include "GRU.h"
#include "LSTM.h"
#include "LayerNorm.h"

#include "Pooling.h"
#include "Reduce.h"
#include "Reorder.h"
#include "Resample.h"
#include "SoftMax.h"
#include "Sum.h"

// Quant
#include "QConv.h"
#include "QDeconv.h"
#include "QMatmul.h"

namespace torch_ipex::xpu::oneDNN {

sycl::event convolution(
    at::Tensor& dst,
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& bia,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    Attr& attr,
    const std::vector<sycl::event>& deps = {});

sycl::event convolution_backward_weights(
    at::Tensor& diff_wgh,
    at::Tensor& diff_bia,
    const at::Tensor& diff_dst,
    const at::Tensor& src,
    IntArrayRef diff_wgh_aten_tz,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    const std::vector<sycl::event>& deps = {});

sycl::event convolution_backward_data(
    at::Tensor& diff_src,
    const at::Tensor& diff_dst,
    const at::Tensor& weight,
    IntArrayRef padding_front_top_left,
    IntArrayRef padding_back_bottom_right,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined,
    const std::vector<sycl::event>& deps = {});

sycl::event matmul(
    Tensor& result,
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& b_raw,
    bool m2_trans,
    torch_ipex::xpu::oneDNN::Attr attr,
    const std::vector<sycl::event>& deps = {});
} // namespace torch_ipex::xpu::oneDNN
