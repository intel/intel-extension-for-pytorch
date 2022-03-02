#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <csrc/dyndisp/DispatchStub.h>

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
    const at::Tensor& bn_scale,
    const at::Tensor& bn_beta,
    const c10::optional<at::Tensor>& bn_weight,
    const c10::optional<at::Tensor>& bn_bias,
    const c10::optional<at::Tensor>& bn_mean,
    const c10::optional<at::Tensor>& bn_var,
    bool bn_training,
    double bn_momentum,
    double bn_eps,
    bool bn_cudnn_enabled,
    int dim);

#if defined(DYN_DISP_BUILD)
namespace {
#endif

at::Tensor concat_bn_relu_kernel_impl(
    const c10::List<at::Tensor>& a,
    const at::Tensor& bn_scale,
    const at::Tensor& bn_beta,
    const c10::optional<at::Tensor>& bn_weight,
    const c10::optional<at::Tensor>& bn_bias,
    const c10::optional<at::Tensor>& bn_mean,
    const c10::optional<at::Tensor>& bn_var,
    bool bn_training,
    double bn_momentum,
    double bn_eps,
    bool bn_cudnn_enabled,
    int dim);

#if defined(DYN_DISP_BUILD)
}
#endif

using concat_bn_relu_kernel_fn = at::Tensor (*)(
    const c10::List<at::Tensor>&,
    const at::Tensor&,
    const at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    bool,
    double,
    double,
    bool,
    int);
DECLARE_DISPATCH(concat_bn_relu_kernel_fn, concat_bn_relu_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
