#pragma once

#include <ATen/Tensor.h>

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <c10/core/Scalar.h>

#include "ideep/ideep.hpp"

namespace torch { namespace jit {

// XXX: PyTorch does not support nesting namespace
// And the alias analysis is not working for namespace other than aten ...
// So we fake some op namespaces to workaround that.
namespace ipex {
  static auto conv2d_base = Symbol::fromQualString("ipex::conv2d_base");
  static auto conv2d_relu = Symbol::fromQualString("ipex::conv2d_relu");
  static auto conv2d_sum = Symbol::fromQualString("ipex::conv2d_sum");
  static auto conv2d_sum_relu = Symbol::fromQualString("ipex::conv2d_sum_relu");

  static auto linear_add = Symbol::fromQualString("ipex::linear_add");
  static auto linear = Symbol::fromQualString("ipex::linear");
  static auto linear_relu = Symbol::fromQualString("ipex::linear_relu");
  static auto linear_gelu = Symbol::fromQualString("ipex::linear_gelu");
  static auto matmul_div = Symbol::fromQualString("ipex::matmul_div");

  // 3d ops
  static auto conv3d_relu = Symbol::fromQualString("ipex::conv3d_relu");
  static auto conv3d_sum = Symbol::fromQualString("ipex::conv3d_sum");
  static auto conv3d_sum_relu = Symbol::fromQualString("ipex::conv3d_sum_relu");

  static auto max_pool2d = Symbol::fromQualString("ipex::max_pool2d");
  static auto softmax = Symbol::fromQualString("ipex::softmax");
  static auto layernorm = Symbol::fromQualString("ipex::layernorm");

  // n-dims tensor op.
  static auto convolution_nd_weight_base =
      Symbol::fromQualString("torch_ipex::convolution_forward");
}

}} // namespace torch::jit

namespace torch_ipex {
namespace cpu {

class AtenIpexJITDev {
 public:
  // for JIT ops
   static at::Tensor
   dil_convolution_base(const at::Tensor &input, const at::Tensor &weight,
                        const at::Tensor &bias, at::IntArrayRef stride,
                        at::IntArrayRef padding, at::IntArrayRef dilation,
                        int64_t groups);

   static at::Tensor
   dil_convolution_swish(const at::Tensor &input, const at::Tensor &weight,
                         const at::Tensor &bias, at::IntArrayRef stride,
                         at::IntArrayRef padding, at::IntArrayRef dilation,
                         int64_t groups);

   static at::Tensor
   dil_convolution_sigmoid(const at::Tensor &input, const at::Tensor &weight,
                           const at::Tensor &bias, at::IntArrayRef stride,
                           at::IntArrayRef padding, at::IntArrayRef dilation,
                           int64_t groups);

   static at::Tensor
   dil_convolution_clamp(const at::Tensor &input, const at::Tensor &weight,
                         const at::Tensor &bias, at::IntArrayRef stride,
                         at::IntArrayRef padding, at::IntArrayRef dilation,
                         int64_t groups, float lower_bound, float upper_bound);

   static at::Tensor
   dil_convolution_relu(const at::Tensor &input, const at::Tensor &weight,
                        const at::Tensor &bias, at::IntArrayRef stride,
                        at::IntArrayRef padding, at::IntArrayRef dilation,
                        int64_t groups);

   static at::Tensor
   dil_convolution_elu(const at::Tensor &input, const at::Tensor &weight,
                       const at::Tensor &bias, at::IntArrayRef stride,
                       at::IntArrayRef padding, at::IntArrayRef dilation,
                       int64_t groups, float alpha, at::Scalar scale,
                       at::Scalar input_scale);

   static at::Tensor &
   dil_convolution_sum(const at::Tensor &input, const at::Tensor &weight,
                       const at::Tensor &bias, at::IntArrayRef stride,
                       at::IntArrayRef padding, at::IntArrayRef dilation,
                       int64_t groups, at::Tensor &accumu, at::Scalar alpha);

   static at::Tensor &
   dil_convolution_sum_relu(const at::Tensor &input, const at::Tensor &weight,
                            const at::Tensor &bias, at::IntArrayRef stride,
                            at::IntArrayRef padding, at::IntArrayRef dilation,
                            int64_t groups, at::Tensor &accumu,
                            at::Scalar alpha);

   static at::Tensor dil_max_pool2d(const at::Tensor &input,
                                    at::IntArrayRef kernel_size,
                                    at::IntArrayRef stride,
                                    at::IntArrayRef padding,
                                    at::IntArrayRef dilation, bool ceil_mode);

   static at::Tensor dil_linear(const at::Tensor &self,
                                const at::Tensor &weight,
                                const at::Tensor &bias);

   static at::Tensor dil_linear_fuse_eltwise(const at::Tensor &self,
                                             const at::Tensor &weight,
                                             const at::Tensor &bias,
                                             const ideep::attr_t &attr);

   static at::Tensor dil_softmax(const at::Tensor &input, const int64_t dim,
                                 const at::IValue &dtype);

   static at::Tensor dil_linear_add(const at::Tensor &self,
                                    const at::Tensor &weight,
                                    const at::Tensor &bias, at::Tensor &accumu,
                                    at::Scalar alpha);

   static at::Tensor dil_matmul_div(const at::Tensor &left,
                                    const at::Tensor &right, at::Tensor out_opt,
                                    const at::Tensor &div_input);

   static at::Tensor dil_matmul_div(const at::Tensor &left,
                                    const at::Tensor &right, at::Tensor out_opt,
                                    const c10::Scalar &div_input);

   static at::Tensor dil_layernorm(const at::Tensor &input,
                                   at::IntArrayRef normalized_shape,
                                   const c10::optional<at::Tensor> &weight_opt,
                                   const c10::optional<at::Tensor> &bias_opt,
                                   float eps, bool cudnn_enable);

   // n-dims tensor op
   static at::Tensor dil_convolution_nd_weight_base(
       const at::Tensor &input, const at::Tensor &weight,
       const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride,
       at::IntArrayRef padding, at::IntArrayRef dilation,
       at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
       bool weight_channels_last, bool weight_prepacked);

   static at::Tensor dil_convolution_nd_weight_swish(
       const at::Tensor &input, const at::Tensor &weight,
       const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride,
       at::IntArrayRef padding, at::IntArrayRef dilation,
       at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
       bool weight_channels_last, bool weight_prepacked);

   static at::Tensor dil_convolution_nd_weight_sigmoid(
       const at::Tensor &input, const at::Tensor &weight,
       const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride,
       at::IntArrayRef padding, at::IntArrayRef dilation,
       at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
       bool weight_channels_last, bool weight_prepacked);

   static at::Tensor dil_convolution_nd_weight_clamp(
       const at::Tensor &input, const at::Tensor &weight,
       const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride,
       at::IntArrayRef padding, at::IntArrayRef dilation,
       at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
       bool weight_channels_last, bool weight_prepacked, float lower_bound,
       float upper_bound);

   static at::Tensor dil_convolution_nd_weight_relu(
       const at::Tensor &input, const at::Tensor &weight,
       const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride,
       at::IntArrayRef padding, at::IntArrayRef dilation,
       at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
       bool weight_channels_last, bool weight_prepacked);

   static at::Tensor dil_convolution_nd_weight_elu(
       const at::Tensor &input, const at::Tensor &weight,
       const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride,
       at::IntArrayRef padding, at::IntArrayRef dilation,
       at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
       bool weight_channels_last, bool weight_prepacked, float alpha,
       at::Scalar scale, at::Scalar input_scale);

   static at::Tensor &dil_convolution_nd_weight_sum(
       const at::Tensor &input, const at::Tensor &weight,
       const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride,
       at::IntArrayRef padding, at::IntArrayRef dilation,
       at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
       bool weight_channels_last, bool weight_prepacked, at::Tensor &accumu,
       at::Scalar alpha);

   static at::Tensor &dil_convolution_nd_weight_sum_relu(
       const at::Tensor &input, const at::Tensor &weight,
       const c10::optional<at::Tensor> &bias_opt, at::IntArrayRef stride,
       at::IntArrayRef padding, at::IntArrayRef dilation,
       at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
       bool weight_channels_last, bool weight_prepacked, at::Tensor &accumu,
       at::Scalar alpha);

   static at::Tensor dil_shuffle(const at::Tensor &self,
                                 at::IntArrayRef view_shape, int64_t dim0,
                                 int64_t dim1);

   // int8 op
   static at::Tensor dil_qembeddingbag(const at::Tensor weight,
                                       const at::Tensor indices,
                                       const at::Tensor offsets, bool sparse,
                                       bool include_last_offset, double w_scale,
                                       int64_t w_zp, at::ScalarType w_dtype,
                                       double o_scale, int64_t o_zp,
                                       at::ScalarType o_dtype);

   static at::Tensor dil_qinteraction(const std::vector<at::Tensor> input,
                                      double o_scale, int64_t o_zp,
                                      at::ScalarType o_dtype);
};

}  // namespace cpu
}  // namespace torch_ipex
