#include "Conv.h"
#include "torch_ipex/csrc/cpu/CustomOPs.h"
#include "torch_ipex/csrc/utils.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

#include "ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

at::Tensor AtenIpexJITDev::dil_convolution_nd_weight_base(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation,
    at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
    bool weight_channels_last, bool weight_prepacked) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_nd_weight_base",
                  std::vector<c10::IValue>({}));
#endif
  return convolution_forward_impl(
      input, weight, bias, stride, padding, dilation, kernel_size, groups,
      output_channel, weight_channels_last, weight_prepacked, ideep::attr_t());
}

at::Tensor AtenIpexJITDev::dil_convolution_nd_weight_swish(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation,
    at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
    bool weight_channels_last, bool weight_prepacked) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_nd_weight_swish",
                  std::vector<c10::IValue>({}));
#endif
  return convolution_forward_impl(input, weight, bias, stride, padding,
                                  dilation, kernel_size, groups, output_channel,
                                  weight_channels_last, weight_prepacked,
                                  ideep::attr_t::fuse_swish());
}

at::Tensor AtenIpexJITDev::dil_convolution_nd_weight_sigmoid(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation,
    at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
    bool weight_channels_last, bool weight_prepacked) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_nd_weight_sigmoid",
                  std::vector<c10::IValue>({}));
#endif
  return convolution_forward_impl(input, weight, bias, stride, padding,
                                  dilation, kernel_size, groups, output_channel,
                                  weight_channels_last, weight_prepacked,
                                  ideep::attr_t::fuse_sigmoid());
}

at::Tensor AtenIpexJITDev::dil_convolution_nd_weight_clamp(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation,
    at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
    bool weight_channels_last, bool weight_prepacked, float lower_bound,
    float upper_bound) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_nd_weight_clamp",
                  std::vector<c10::IValue>({}));
#endif
  return convolution_forward_impl(
      input, weight, bias, stride, padding, dilation, kernel_size, groups,
      output_channel, weight_channels_last, weight_prepacked,
      ideep::attr_t::fuse_clamp(lower_bound, upper_bound));
}

at::Tensor AtenIpexJITDev::dil_convolution_nd_weight_relu(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation,
    at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
    bool weight_channels_last, bool weight_prepacked) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_nd_weight_relu",
                  std::vector<c10::IValue>({}));
#endif
  return convolution_forward_impl(input, weight, bias, stride, padding,
                                  dilation, kernel_size, groups, output_channel,
                                  weight_channels_last, weight_prepacked,
                                  ideep::attr_t::fuse_relu());
}

at::Tensor AtenIpexJITDev::dil_convolution_nd_weight_elu(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation,
    at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
    bool weight_channels_last, bool weight_prepacked, float alpha,
    at::Scalar scale, at::Scalar input_scale) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_nd_weight_elu",
                  std::vector<c10::IValue>({}));
#endif
  auto scale_value = scale.to<float>();
  auto input_scale_value = input_scale.to<float>();
  return convolution_forward_impl(
      input, weight, bias, stride, padding, dilation, kernel_size, groups,
      output_channel, weight_channels_last, weight_prepacked,
      ideep::attr_t::fuse_elu(scale_value, alpha, input_scale_value));
}

at::Tensor &AtenIpexJITDev::dil_convolution_nd_weight_sum(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation,
    at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
    bool weight_channels_last, bool weight_prepacked, at::Tensor &accumu,
    at::Scalar alpha) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_nd_weight_sum",
                  std::vector<c10::IValue>({}));
#endif
  auto scale = alpha.to<float>();
  convolution_forward_inplace_impl(
      input, weight, bias, accumu, stride, padding, dilation, kernel_size,
      groups, output_channel, weight_channels_last, weight_prepacked,
      ideep::attr_t::fuse_sum(scale));
  return accumu;
}

at::Tensor &AtenIpexJITDev::dil_convolution_nd_weight_sum_relu(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation,
    at::IntArrayRef kernel_size, int64_t groups, int64_t output_channel,
    bool weight_channels_last, bool weight_prepacked, at::Tensor &accumu,
    at::Scalar alpha) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_nd_weight_sum_relu",
                  std::vector<c10::IValue>({}));
#endif
  auto scale = alpha.to<float>();
  convolution_forward_inplace_impl(
      input, weight, bias, accumu, stride, padding, dilation, kernel_size,
      groups, output_channel, weight_channels_last, weight_prepacked,
      ideep::attr_t::residual(scale));
  return accumu;
}

} // namespace cpu
} // namespace torch_ipex
