#include "torch_ipex/csrc/cpu/FusionOPs.h"

#include <ATen/Context.h>
#include <ATen/CPUGenerator.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

#include <limits>

#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/ipex_tensor_impl.h"
#include "torch_ipex/csrc/utils.h"
#include "dbl/Common.h"
#include "dbl/Conv.h"
#include "dbl/Linear.h"
#include "ShadeDataContext.h"

#include "dil/dil.hpp"

namespace torch_ipex {
namespace cpu {

using namespace dbl::comm;

at::Tensor dil_convolution_outplace_fusion(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const dil::attr_t& op_attr) {
  dil::tensor dil_input;
  dil::tensor dil_weight;
  c10::optional<dil::tensor> dil_bias{c10::nullopt};

  auto input_contiguous = input.is_contiguous() ? input : input.contiguous();
  auto weight_contiguous = weight.is_contiguous() ? weight : weight.contiguous();

  reorder_to_bf16_for_mix_prec(input_contiguous);
  dil_input = try_gen_dil_tensor(input_contiguous);

  if (bias.defined()) {
    auto bias_contiguous = bias.is_contiguous() ? bias : bias.contiguous();
    reorder_to_bf16_for_mix_prec(bias_contiguous);
    dil_bias = try_gen_dil_tensor(bias_contiguous);
  }

  reorder_to_bf16_for_mix_prec(weight_contiguous);
  dbl::conv::prepack_conv_weights(
    input_contiguous,
    dil_input,
    weight_contiguous,
    stride,
    padding,
    dilation,
    groups);
  dil_weight = try_gen_dil_tensor(weight_contiguous);

  dil::tensor dil_output = dbl::conv::convolution_impl(
    dil_input,
    dil_weight,
    dil_bias,
    padding,
    stride,
    dilation,
    groups,
    op_attr);

  return gen_aten_tensor_by(std::move(dil_output));
}

static at::Tensor& dil_convolution_inplace_fusion(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& accumu,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const dil::attr_t& attr) {
  dil::tensor dil_input;
  dil::tensor dil_weight;
  dil::tensor dil_output;
  c10::optional<dil::tensor> dil_bias{c10::nullopt};

  auto input_contiguous =  input.is_contiguous() ? input : input.contiguous();
  auto weight_contiguous = weight.is_contiguous() ? weight : weight.contiguous();
  auto output_contiguous = accumu.is_contiguous() ? accumu : accumu.contiguous();

  reorder_to_bf16_for_mix_prec(input_contiguous);
  reorder_to_bf16_for_mix_prec(output_contiguous);
  dil_input = try_gen_dil_tensor(input_contiguous);
  dil_output = try_gen_dil_tensor(output_contiguous);

  if (bias.defined()) {
    auto bias_contiguous = bias.is_contiguous() ? bias : bias.contiguous();
    reorder_to_bf16_for_mix_prec(bias_contiguous);
    dil_bias = try_gen_dil_tensor(bias_contiguous);
  }

  reorder_to_bf16_for_mix_prec(weight_contiguous);
  dbl::conv::prepack_conv_weights(
    input_contiguous,
    dil_input,
    weight_contiguous,
    stride,
    padding,
    dilation,
    groups);
  dil_weight = try_gen_dil_tensor(weight_contiguous);

  dbl::conv::convolution_inplace_impl(
    dil_input,
    dil_weight,
    dil_bias,
    dil_output,
    padding,
    stride,
    dilation,
    groups,
    attr);

  sync_shape_from_dil_to_aten(accumu, dil_output);
  return accumu;
}

at::Tensor AtenIpexJITDev::dil_convolution_swish(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  return dil_convolution_outplace_fusion(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    dil::attr_t::fuse_swish());
}

at::Tensor AtenIpexJITDev::dil_convolution_sigmoid(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  return dil_convolution_outplace_fusion(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    dil::attr_t::fuse_sigmoid());
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
  return dil_convolution_outplace_fusion(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    dil::attr_t::fuse_clamp(lower_bound, upper_bound));
}

at::Tensor AtenIpexJITDev::dil_convolution_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  return dil_convolution_outplace_fusion(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    dil::attr_t::fuse_relu());
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
    dil::attr_t::fuse_elu(scale_value, alpha, input_scale_value));
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
    dil::attr_t::fuse_sum(scale));
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
    dil::attr_t::residual(scale));
}

at::Tensor AtenIpexJITDev::dil_linear_fuse_relu(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  IPEX_CHECK(self.dim() >= 2,
      "dil_linear: input needs to has dim at least 2, input dim ", self.dim());
  auto input_contiguous = self.is_contiguous() ? self : self.contiguous();
  auto weight_contiguous = weight.is_contiguous() ? weight : weight.contiguous();

  reorder_to_bf16_for_mix_prec(input_contiguous);
  reorder_to_bf16_for_mix_prec(weight_contiguous);

  // reshape first if input dim is greater than 2 and the reshape will cost a memory copy.
  auto self_reshaped = self.dim() > 2 ? self.reshape({-1, input_contiguous.size(self.dim() - 1)}) : self;
  const dil::tensor x = try_gen_dil_tensor(self_reshaped);
  const dil::tensor w = try_gen_dil_tensor(weight_contiguous);

  c10::optional<dil::tensor> b{c10::nullopt};
  if (bias.defined()) {
    auto bias_contiguous = bias.is_contiguous() ? bias : bias.contiguous();
    reorder_to_bf16_for_mix_prec(bias_contiguous);
    b = try_gen_dil_tensor(bias_contiguous);
  }

  dil::tensor y = dbl::linear::linear_impl(x, w, b, /* dst_scale */ dil::scale_t(), dil::attr_t::fuse_relu());

  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));

  if (self.dim() > 2) {
    return gen_aten_tensor_by(std::move(y)).reshape(output_size);
  }
  return gen_aten_tensor_by(std::move(y));
}

}  // namespace cpu
}  // namespace torch_ipex
