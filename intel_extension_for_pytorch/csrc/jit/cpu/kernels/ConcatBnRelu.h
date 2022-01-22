#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>

namespace torch_ipex {
namespace cpu {

/**
 * This operator fuses Concat + BN + ReLU specifically for the tensors with the
 * same sizes. Please refer
 * https://github.com/soCzech/TransNetV2/blob/master/inference-pytorch/transnetv2_pytorch.py
 * for the graph structure.
 * */
at::Tensor ConcatBnRelu(
    const c10::List<at::Tensor>& a,
    const at::Tensor& bn_beta,
    const c10::optional<at::Tensor>& bn_scale,
    const c10::optional<at::Tensor>& bn_bias,
    const c10::optional<at::Tensor>& bn_mean,
    const c10::optional<at::Tensor>& bn_var,
    bool bn_training,
    double bn_momentum,
    double bn_eps,
    bool bn_cudnn_enabled,
    int dim);

} // namespace cpu
} // namespace torch_ipex
