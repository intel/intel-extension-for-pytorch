#include "torch_ipex/csrc/cpu/FusionOPs.h"
#include "torch_ipex/csrc/utils.h"
#include "Conv.h"
#include "Linear.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

#include <ideep.hpp>

namespace torch_ipex {
namespace cpu {

at::Tensor dil_convolution_outplace_fusion(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& op_attr,
    const std::string& op_name) {

  at::Tensor output = convolution_impl(
    IS_CONTIGUOUS_ANY(input) ? input : input.contiguous(),
    IS_CONTIGUOUS_ANY(weight) ? weight : weight.contiguous(),
    (!bias.defined()) || IS_CONTIGUOUS_ANY(bias) ? bias : bias.contiguous(),
    stride,
    padding,
    dilation,
    groups,
    op_attr);

  return output;
}

static at::Tensor& dil_convolution_inplace_fusion(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const ideep::attr_t& attr,
    const std::string& op_name) {

  convolution_inplace_impl(
    IS_CONTIGUOUS_ANY(input) ? input : input.contiguous(),
    IS_CONTIGUOUS_ANY(weight) ? weight : weight.contiguous(),
    (!bias.defined()) || IS_CONTIGUOUS_ANY(bias) ? bias : bias.contiguous(),
    output,
    stride,
    padding,
    dilation,
    groups,
    attr);

  return output;
}

at::Tensor AtenIpexJITDev::dil_convolution_swish(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_swish", std::vector<c10::IValue>({input, weight, bias}), torch::autograd::Node::peek_at_next_sequence_nr());
#endif
  return dil_convolution_outplace_fusion(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    ideep::attr_t::fuse_swish(),
    "convolution_swish");
}

at::Tensor AtenIpexJITDev::dil_convolution_sigmoid(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_sigmoid", std::vector<c10::IValue>({input, weight, bias}), torch::autograd::Node::peek_at_next_sequence_nr());
#endif
  return dil_convolution_outplace_fusion(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    ideep::attr_t::fuse_sigmoid(),
    "convolution_sigmoid");
}

at::Tensor AtenIpexJITDev::dil_convolution_clamp(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    float lower_bound,
    float upper_bound) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_clamp", std::vector<c10::IValue>({input, weight, bias}), torch::autograd::Node::peek_at_next_sequence_nr());
#endif
  return dil_convolution_outplace_fusion(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    ideep::attr_t::fuse_clamp(lower_bound, upper_bound),
    "convolution_clamp");
}

at::Tensor AtenIpexJITDev::dil_convolution_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_relu", std::vector<c10::IValue>({input, weight, bias}), torch::autograd::Node::peek_at_next_sequence_nr());
#endif
  return dil_convolution_outplace_fusion(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    ideep::attr_t::fuse_relu(),
    "Convolution_Relu");
}

at::Tensor AtenIpexJITDev::dil_convolution_elu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    float alpha,
    at::Scalar scale,
    at::Scalar input_scale) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_elu", std::vector<c10::IValue>({input, weight, bias}), torch::autograd::Node::peek_at_next_sequence_nr());
#endif
  auto scale_value = scale.to<float>();
  auto input_scale_value = input_scale.to<float>();
  return dil_convolution_outplace_fusion(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    ideep::attr_t::fuse_elu(scale_value, alpha, input_scale_value),
    "convolution_elu");
}

at::Tensor& AtenIpexJITDev::dil_convolution_sum(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    at::Tensor& accumu,
    at::Scalar alpha) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_sum", std::vector<c10::IValue>({input, weight, bias}), torch::autograd::Node::peek_at_next_sequence_nr());
#endif
  auto scale = alpha.to<float>();
  return dil_convolution_inplace_fusion(
    input,
    weight,
    bias,
    accumu,
    stride,
    padding,
    dilation,
    groups,
    ideep::attr_t::fuse_sum(scale),
    "Convolution_Sum");
}

at::Tensor& AtenIpexJITDev::dil_convolution_sum_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    at::Tensor& accumu,
    at::Scalar alpha) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_sum_relu", std::vector<c10::IValue>({input, weight, bias}), torch::autograd::Node::peek_at_next_sequence_nr());
#endif
  auto scale = alpha.to<float>();
  return dil_convolution_inplace_fusion(
    input,
    weight,
    bias,
    accumu,
    stride,
    padding,
    dilation,
    groups,
    ideep::attr_t::residual(scale),
    "Convolution_Sum_Relu");
}

at::Tensor AtenIpexJITDev::dil_linear_fuse_eltwise(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const ideep::attr_t& attr) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_linear_fuse_eltwise", std::vector<c10::IValue>({self, weight, bias}), torch::autograd::Node::peek_at_next_sequence_nr());
#endif
  return linear_impl(
    self.is_contiguous() ? self : self.contiguous(),
    weight.is_contiguous() ? weight : weight.contiguous(),
    (!bias.defined()) || bias.is_contiguous() ? bias : bias.contiguous(),
    attr);
}

}  // namespace cpu
}  // namespace torch_ipex
