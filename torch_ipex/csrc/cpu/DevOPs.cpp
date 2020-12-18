#include "torch_ipex/csrc/cpu/DevOPs.h"

#include <ATen/Context.h>
#include <ATen/CPUGenerator.h>
#include <ATen/InferSize.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/record_function.h>

#include <limits>

#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/ipex_tensor_impl.h"
#include "torch_ipex/csrc/utils.h"
#include "bf16/DevOPs.hpp"
#include "dbl/Common.h"
#include "dbl/Conv.h"
#include "dbl/Deconv.h"
#include "dbl/Pool.h"
#include "dbl/DNNLChecker.h"
#include "dbl/Linear.h"
#include "dbl/RNN.h"
#include "dbl/UpSample.h"
#include "ShadeDataContext.h"

#include "dil/dil.hpp"

namespace torch_ipex {
namespace cpu {

#if defined(IPEX_DISP_OP)
#define DEBUG(fmt) printf(fmt);
#else
#define DEBUG(fmt)
#endif

#define CHECK_DNNL_OP_PRE_COND(tensor)                                               \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.defined());                                \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(tensor.layout() == c10::kStrided)

at::Tensor AtenIpexCPUDev::dil_convolution(
    const at::Tensor & input,
    const at::Tensor & weight,
    const at::Tensor & bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  DEBUG("AtenIpexCPUDev::dil_convolution\n");
  dil::tensor dil_input;
  dil::tensor dil_weight;
  c10::optional<dil::tensor> dil_bias{c10::nullopt};

  CHECK_DNNL_OP_PRE_COND(input);
  CHECK_DNNL_OP_PRE_COND(weight);

  std::vector<float> output_scale = {};
  bool quantized = false;
  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<std::vector<float>> scales;
    std::tie(scales, quantized) = dbl::comm::get_int8_scales({input}, /* uint8_used for output*/false);
    //quantized = false;
    if (quantized) {
      output_scale.push_back(scales[1][0]);
      dbl::comm::reorder_to_int8_for_mix_prec(input, scales[0]);
      dbl::comm::reorder_to_int8_for_mix_prec(weight, {});
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
      dbl::comm::reorder_to_dtype(weight, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input);
    dbl::comm::reorder_to_bf16_for_mix_prec(weight, true);
  }

  dil_input = dbl::comm::try_gen_dil_tensor(input);
  // In the case of the auto mix precision, do not prepack
  // the weight during the training
  if (!(check_auto_mix_bf16_fp32() && check_train())) {
    dbl::conv::prepack_conv_weights(input, dil_input,
      weight, stride, padding, dilation, groups);
  }

  dil_weight = dbl::comm::try_gen_dil_tensor(weight);

  if (bias.defined()) {
    CHECK_DNNL_OP_PRE_COND(bias);
    if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
      if (quantized) {
        auto src = dbl::comm::try_gen_dil_storage(bias);
        auto src_type = src.get_data_type();
        if (src_type != dil::data_type::s32) {
          auto dst_desc = src.get_desc().to_type(dil::data_type::s32);
          auto bias_scales = dil_weight.get_scale();
          for (auto &scale : bias_scales) { scale *= dil_input.get_scale()[0];  }
          dbl::comm::reorder_to_desc(bias, dst_desc, bias_scales);
        }
      } else {
        dbl::comm::reorder_to_dtype(bias, at::kFloat);
      }
    } else {
      dbl::comm::reorder_to_bf16_for_mix_prec(bias, true);
    }
    dil_bias = dbl::comm::try_gen_dil_tensor(bias);
  }

  dil::tensor dil_output = dbl::conv::convolution_impl(
    dil_input,
    dil_weight,
    dil_bias,
    padding,
    stride,
    dilation,
    groups,
    dil::attr_t(),
    output_scale);

  auto aten_output = dbl::comm::gen_aten_tensor_by(std::move(dil_output));

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer({input}, {aten_output}, "Convolution");
  }

  return aten_output;
}

at::Tensor dil_convolution_backward_input(
    at::IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  // for training case, grad_output can be cpu tensor or MKLDNN tensor,
  // but weight and bias always cpu tensor
  auto dil_grad_output = dbl::comm::try_gen_dil_tensor(grad_output);
  auto dil_weight = dbl::comm::try_gen_dil_tensor(weight);

  dil::tensor dil_grad_input;
  dil::convolution_backward_data::compute(
      dil_grad_output,
      dil_weight,
      input_size.vec(),
      dil_grad_input,
      stride.vec(),
      dilation.vec(),
      padding.vec(),
      padding.vec(),
      groups);
  return dbl::comm::gen_aten_tensor_by(std::move(dil_grad_input));
}

std::tuple<at::Tensor, at::Tensor> dil_convolution_backward_weights(
    const at::Tensor& weight, const at::Tensor& grad_output, const at::Tensor& input,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  // for training case, grad_output and input can be cpu tensor or MKLDNN tensor,
  // but weight and bias always cpu tensor
  const dil::tensor dil_grad_output = dbl::comm::try_gen_dil_tensor(grad_output);
  const dil::tensor dil_input = dbl::comm::try_gen_dil_tensor(input);

  dil::tensor dil_grad_weight, dil_grad_bias;
  dil::tensor w = dbl::comm::try_gen_dil_tensor(weight);
  auto diff_weight_type = w.get_data_type();
  auto weight_size = weight.sizes();

  if (bias_defined) {
    dil::convolution_backward_weights::compute(
        dil_input,
        dil_grad_output,
        weight_size.vec(),
        dil_grad_weight,
        dil_grad_bias,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups,
        diff_weight_type);
    return std::make_tuple(
        dbl::comm::gen_aten_tensor_by(std::move(dil_grad_weight)),
        dbl::comm::gen_aten_tensor_by(std::move(dil_grad_bias)));
  } else {
    dil::convolution_backward_weights::compute(
        dil_input,
        dil_grad_output,
        weight_size.vec(),
        dil_grad_weight,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups,
        diff_weight_type);
    return std::make_tuple(
        dbl::comm::gen_aten_tensor_by(std::move(dil_grad_weight)),
        at::Tensor());
  }
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> AtenIpexCPUDev::dil_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask)
{
  DEBUG("AtenIpexCPUDev::dil_convolution_backward\n");
  at::Tensor grad_output = grad_output_t.is_contiguous() ? grad_output_t : grad_output_t.contiguous();
  CHECK_DNNL_OP_PRE_COND(input);
  CHECK_DNNL_OP_PRE_COND(weight);
  dbl::comm::reorder_to_bf16_for_mix_prec(input);
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dbl::comm::reorder_to_bf16_for_mix_prec(weight, true);

  at::Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = dil_convolution_backward_input(
      input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = dil_convolution_backward_weights(
      weight, grad_output, input, padding, stride, dilation, groups, output_mask[2]);
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

at::Tensor AtenIpexCPUDev::dil_deconvolution(
    const at::Tensor & input,
    const at::Tensor & weight,
    const at::Tensor & bias,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups) {
  DEBUG("AtenIpexCPUDev::dil_deconvolution\n");
  dil::tensor dil_input;
  dil::tensor dil_weight;
  c10::optional<dil::tensor> dil_bias{c10::nullopt};

  CHECK_DNNL_OP_PRE_COND(input);
  CHECK_DNNL_OP_PRE_COND(weight);

  dbl::comm::reorder_to_bf16_for_mix_prec(input);
  dil_input = dbl::comm::try_gen_dil_tensor(input);

  if (bias.defined()) {
    CHECK_DNNL_OP_PRE_COND(bias);
    dbl::comm::reorder_to_bf16_for_mix_prec(bias, true);
    dil_bias = dbl::comm::try_gen_dil_tensor(bias);
  }

  dbl::comm::reorder_to_bf16_for_mix_prec(weight, true);

  std::vector<int64_t> padding_r = dbl::deconv::calc_padding_r_adjusted(input.dim(), padding, output_padding);

  if (!(check_auto_mix_bf16_fp32() && check_train())) {
    dbl::deconv::prepack_deconv_weights(
      input, weight, stride, padding, padding_r, output_padding, dilation, groups, bias.defined());
  }
  dil_weight = dbl::comm::try_gen_dil_tensor(weight);

  dil::tensor dil_output = dbl::deconv::deconvolution_impl(
    dil_input,
    dil_weight,
    dil_bias,
    padding,
    padding_r,
    output_padding,
    stride,
    dilation,
    groups);

  return dbl::comm::gen_aten_tensor_by(std::move(dil_output));
}

at::Tensor dil_deconvolution_backward_input(
    at::IntArrayRef input_size, 
    const at::Tensor& grad_output, 
    const at::Tensor& weight,
    at::IntArrayRef padding,
    std::vector<int64_t> padding_r,
    at::IntArrayRef stride, 
    at::IntArrayRef dilation, 
    int64_t groups, 
    bool bias_defined) {
      // for training case, grad_output can be cpu tensor or MKLDNN tensor,
      // but weight and bias always cpu tensor
      auto dil_grad_output = dbl::comm::try_gen_dil_tensor(grad_output);
      auto dil_weight = dbl::comm::try_gen_dil_tensor(weight);

      dil::tensor dil_grad_input;
      dil::convolution_transpose_backward_data::compute(
          dil_grad_output,
          dil_weight,
          input_size.vec(),
          dil_grad_input,
          stride.vec(),
          padding.vec(),
          padding_r,
          dilation.vec(),
          groups);
      return dbl::comm::gen_aten_tensor_by(std::move(dil_grad_input));
}

std::tuple<at::Tensor, at::Tensor> dil_deconvolution_backward_weights(
    const at::Tensor& weight, 
    const at::Tensor& grad_output, 
    const at::Tensor& input,
    at::IntArrayRef padding,
    std::vector<int64_t> padding_r,
    at::IntArrayRef stride, 
    at::IntArrayRef dilation, 
    int64_t groups, 
    bool bias_defined) { 
      // for training case, grad_output and input can be cpu tensor or MKLDNN tensor,
      // but weight and bias always cpu tensor
      const dil::tensor dil_grad_output = dbl::comm::try_gen_dil_tensor(grad_output);
      const dil::tensor dil_input = dbl::comm::try_gen_dil_tensor(input);

      dil::tensor dil_grad_weight, dil_grad_bias;
      dil::tensor w = dbl::comm::try_gen_dil_tensor(weight);
      auto diff_weight_type = w.get_data_type();
      auto weight_size = weight.sizes();

      if (bias_defined) {
        dil::convolution_transpose_backward_weights::compute(
            dil_input,
            dil_grad_output,
            weight_size.vec(),
            dil_grad_weight,
            dil_grad_bias,
            stride.vec(),
            padding.vec(),
            padding_r,
            dilation.vec(),
            groups,
            diff_weight_type);
        return std::make_tuple(
            dbl::comm::gen_aten_tensor_by(std::move(dil_grad_weight)),
            dbl::comm::gen_aten_tensor_by(std::move(dil_grad_bias)));
      } else {
        dil::convolution_transpose_backward_weights::compute(
            dil_input,
            dil_grad_output,
            weight_size.vec(),
            dil_grad_weight,
            stride.vec(),
            padding.vec(),
            padding_r,
            dilation.vec(),
            groups, 
            diff_weight_type);
        return std::make_tuple(
            dbl::comm::gen_aten_tensor_by(std::move(dil_grad_weight)),
            at::Tensor());
      }
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> AtenIpexCPUDev::dil_deconvolution_backward(
  const at::Tensor& input, 
  const at::Tensor& grad_output, 
  const at::Tensor& weight, 
  at::IntArrayRef padding, 
  at::IntArrayRef output_padding,
  at::IntArrayRef stride, 
  at::IntArrayRef dilation, 
  int64_t groups, 
  std::array<bool,3> output_mask) {
    DEBUG("AtenIpexCPUDev::dil_deconvolution_backward\n");
    CHECK_DNNL_OP_PRE_COND(input);
    CHECK_DNNL_OP_PRE_COND(weight);
    dbl::comm::reorder_to_bf16_for_mix_prec(input);
    dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
    dbl::comm::reorder_to_bf16_for_mix_prec(weight, true);

    // adjust padding_r in deconvolution
    std::vector<int64_t> padding_r = dbl::deconv::calc_padding_r_adjusted(input.dim(), padding, output_padding);

    at::Tensor grad_input, grad_weight, grad_bias;
    if (output_mask[0]) {
      grad_input = dil_deconvolution_backward_input(
        input.sizes(), grad_output, weight, padding, padding_r, stride, dilation, groups, output_mask[2]);
    }
    if (output_mask[1] || output_mask[2]) {
      std::tie(grad_weight, grad_bias) = dil_deconvolution_backward_weights(
        weight, grad_output, input, padding, padding_r, stride, dilation, groups, output_mask[2]);
    }
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

at::Tensor AtenIpexCPUDev::dil_convolution_overrideable(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  DEBUG("AtenIpexCPUDev::convolution_overrideable\n");
  try {
    if (check_auto_dnnl()) {
      std::vector<at::Tensor> dnnl_input_tensors;
      dnnl_input_tensors.push_back(input);
      dnnl_input_tensors.push_back(weight);
      if (bias.defined()) {
        dnnl_input_tensors.push_back(bias);
      }
      if (dbl::chk::dnnl_support_the_tensors(dnnl_input_tensors)) {
        if (transposed) {
          return AtenIpexCPUDev::dil_deconvolution(input.is_contiguous() ? input : input.contiguous(), weight.is_contiguous() ? weight : weight.contiguous(), bias.defined() && !bias.is_contiguous() ? bias.contiguous() : bias, padding, output_padding, stride, dilation, groups);
        } else {
          // for int8 path, input always acbd format which is non-contiguous, .contiguous() will reorder to fp32
          auto src_dil_type = dbl::comm::try_gen_dil_tensor(input).get_data_type();
          auto input_temp = (src_dil_type == dil::data_type::u8 || src_dil_type == dil::data_type::s8 || input.is_contiguous()) ? input : input.contiguous();
          auto weight_dil_type = dbl::comm::try_gen_dil_tensor(weight).get_data_type();
          auto weight_temp = (weight_dil_type == dil::data_type::s8 || weight.is_contiguous()) ? weight : weight.contiguous();
          return AtenIpexCPUDev::dil_convolution(input_temp, weight_temp, bias, stride, padding, dilation, groups);
        }
      }
    }
  } catch (std::exception& e) {
#if defined(_DEBUG)
    TORCH_WARN(e.what());
#endif
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weight.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(bias.layout() == c10::kStrided);
  auto&& _ipex_input = bridge::shallowFallbackToCPUTensor(input);
  auto&& _ipex_weight = bridge::shallowFallbackToCPUTensor(weight);
  auto&& _ipex_bias = bridge::shallowFallbackToCPUTensor(bias);
  auto&& _ipex_result = at::convolution(_ipex_input, _ipex_weight, _ipex_bias, stride, padding, dilation, transposed, output_padding, groups);
  static_cast<void>(_ipex_result); // Avoid warnings in case not used
  return bridge::shallowUpgradeToDPCPPTensor(_ipex_result);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> AtenIpexCPUDev::dil_convolution_backward_overrideable(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, std::array<bool,3> output_mask) {
  DEBUG("AtenIpexCPUDev::convolution_backward_overrideable\n");
  try {
    if (check_auto_dnnl()) {
      // NOTE: DO NOT always call contiguous. It may break lazy-reorder. Because contiguous will call reorder instantly.
      std::vector<at::Tensor> dnnl_input_tensors;
      dnnl_input_tensors.push_back(input);
      dnnl_input_tensors.push_back(weight);
      dnnl_input_tensors.push_back(grad_output);
      if (dbl::chk::dnnl_support_the_tensors(dnnl_input_tensors)) {
        if (transposed) {
          return AtenIpexCPUDev::dil_deconvolution_backward(
            input.is_contiguous() ? input : input.contiguous(),
            grad_output.is_contiguous() ? grad_output : grad_output.contiguous(),
            weight.is_contiguous() ? weight : weight.contiguous(),
            padding,
            output_padding,
            stride,
            dilation,
            groups,
            output_mask);
        } else {
          return AtenIpexCPUDev::dil_convolution_backward(
            input.is_contiguous() ? input : input.contiguous(),
            grad_output.is_contiguous() ? grad_output : grad_output.contiguous(),
            weight.is_contiguous() ? weight : weight.contiguous(),
            padding,
            stride,
            dilation,
            groups,
            output_mask);
        }
      }
    }
  } catch (std::exception& e) {
#if defined(_DEBUG)
    TORCH_WARN(e.what());
#endif 
  }

  if (transposed) {
    return AtenIpexCPUDev::cpu_deconvolution_backward(input, grad_output, weight, padding, output_padding, stride, dilation, groups, output_mask);
  } else {
    // TODO should fallback to cpu (maybe thnn 2d or thnn 3d conv bw) rather than mkldnn
    return AtenIpexCPUDev::mkldnn_convolution_backward(input, grad_output, weight, padding, stride, dilation, groups, output_mask);
  }
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> AtenIpexCPUDev::mkldnn_convolution_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  DEBUG("AtenIpexCPUDev::mkldnn_convolution_backward\n");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.defined());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_output.defined());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weight.defined());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_output.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weight.layout() == c10::kStrided);
  auto&& _ipex_self = bridge::shallowFallbackToCPUTensor(self);
  auto&& _ipex_grad_output = bridge::shallowFallbackToCPUTensor(grad_output);
  auto&& _ipex_weight = bridge::shallowFallbackToCPUTensor(weight);
  auto&& _ipex_result = at::mkldnn_convolution_backward(_ipex_self.contiguous(), _ipex_grad_output.contiguous(), _ipex_weight.contiguous(), padding, stride, dilation, groups, output_mask);
  static_cast<void>(_ipex_result); // Avoid warnings in case not used
  return std::tuple<at::Tensor,at::Tensor,at::Tensor>(bridge::shallowUpgradeToDPCPPTensor(std::get<0>(_ipex_result)), bridge::shallowUpgradeToDPCPPTensor(std::get<1>(_ipex_result)), bridge::shallowUpgradeToDPCPPTensor(std::get<2>(_ipex_result)));
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> AtenIpexCPUDev::cpu_deconvolution_backward(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  DEBUG("AtenIpexCPUDev::cpu_deconvolution_backward\n");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.defined());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_output.defined());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weight.defined());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grad_output.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weight.layout() == c10::kStrided);
  auto&& _ipex_self = bridge::shallowFallbackToCPUTensor(self);
  auto&& _ipex_grad_output = bridge::shallowFallbackToCPUTensor(grad_output);
  auto&& _ipex_weight = bridge::shallowFallbackToCPUTensor(weight);

  /*
    when groups != 1, we should:
        1). slice _ipex_grad_output ,_ipex_weight and _ipex_self according to groups,;
        2). call at::slow_conv_transpose2d_backward or at::slow_conv_transpose3d_backward on the sliced tensors 
            (since these two functions only support groups = 1) to calculate g_input, g_weight and g_bias
        3). cat g_input, g_weight and g_bias in each group and return

      In cpu path, decovolution_forward will do the following:
  
      aten/src/ATen/native/Convolution.cpp:
      
        std::vector<Tensor> outputs(params.groups);
        input = input.contiguous();
        for (int g = 0; g < params.groups; ++g) {
          auto input_g = subtensor(input, 1, params.groups, g);
          auto weight_g = subtensor(weight, 0, params.groups, g);
          auto bias_g = subtensor(bias, 0, params.groups, g);
          outputs[g] = at::_convolution_nogroup(
              input_g, weight_g, bias_g, params.stride, params.padding, params.dilation, params.transposed, params.output_padding);
        }
        output = at::cat(outputs, 1);

      Example:
        groups: 2
        input: [2, 10, 8, 8]
        kernel: [10, 50, 3, 3]

        conv bw weight: [10, 25, 3, 3]
        conv bw input: [2, 10, 8, 8]
        conv bw grad_output: [2, 50, 14, 14]
  */

  auto dim = self.ndimension();
  auto kernel_size = weight.sizes().slice(2);

  IPEX_CHECK(dim == 4 || dim == 5, "deconvolution backward fallback only support 2d or 3d deconv");

  at::Tensor g_input_concat, g_weight_concat, g_bias_concat;
  
  std::vector<at::Tensor> g_input(groups), g_weight(groups), g_bias(groups);
  
  _ipex_self = _ipex_self.is_contiguous() ? _ipex_self : _ipex_self.contiguous();
  _ipex_grad_output = _ipex_grad_output.is_contiguous() ? _ipex_grad_output : _ipex_grad_output.contiguous();
  _ipex_weight = _ipex_weight.is_contiguous() ? _ipex_weight : _ipex_weight.contiguous();
  for (int g = 0; g < groups; ++g) {
    auto _ipex_self_g = dbl::comm::subtensor(_ipex_self, 1, groups, g);
    auto _ipex_grad_output_g = dbl::comm::subtensor(_ipex_grad_output, 1, groups, g);
    auto _ipex_weight_g = dbl::comm::subtensor(_ipex_weight, 0, groups, g);
    
    if (dim == 4) {
      std::tie(g_input[g], g_weight[g], g_bias[g]) = at::slow_conv_transpose2d_backward(
        _ipex_grad_output_g, 
        _ipex_self_g, 
        _ipex_weight_g, 
        kernel_size, 
        stride, 
        padding, 
        output_padding, 
        dilation, 
        empty_like(_ipex_grad_output_g, at::MemoryFormat::Contiguous), 
        empty_like(_ipex_grad_output_g, at::MemoryFormat::Contiguous), 
        output_mask);
    } else {
      std::tie(g_input[g], g_weight[g], g_bias[g]) = at::slow_conv_transpose3d_backward(
        _ipex_grad_output_g,
        _ipex_self_g,
        _ipex_weight_g, 
        kernel_size, 
        stride, 
        padding, 
        output_padding, 
        dilation, 
        empty_like(_ipex_grad_output_g, at::MemoryFormat::Preserve), 
        empty_like(_ipex_grad_output_g, at::MemoryFormat::Preserve), 
        output_mask);
    }
  }
  g_input_concat = at::cat(g_input, 1);
  g_weight_concat = at::cat(g_weight, 0);
  g_bias_concat = output_mask[2] ? at::cat(g_bias, 0) : at::Tensor();

  return std::tuple<at::Tensor,at::Tensor,at::Tensor>(bridge::shallowUpgradeToDPCPPTensor(g_input_concat), bridge::shallowUpgradeToDPCPPTensor(g_weight_concat), bridge::shallowUpgradeToDPCPPTensor(g_bias_concat));
}

template<bool inplace>
at::Tensor& dil_add_common(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Scalar alpha) {
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(other);

  IPEX_CHECK(self.sizes().equals(other.sizes()),
      "dil add not support broadcast yet");
  if (check_auto_mix_int8_fp32()) {
    // for accuracy, reorder int8 to fp32
    dbl::comm::reorder_to_dtype(self, at::kFloat);
    dbl::comm::reorder_to_dtype(other, at::kFloat);
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(self, true);
    dbl::comm::reorder_to_bf16_for_mix_prec(other, true);
  }

  auto x = dbl::comm::try_gen_dil_tensor(self);
  auto y_ = dbl::comm::try_gen_dil_tensor(other);
  // reorder other to the data type of self
  auto y = dbl::comm::reorder_dil_tensor_to_dtype(y_, x.get_data_type());
  auto z = inplace ? x : dil::tensor();

  dil::sum::compute({1.0, alpha.to<float>()}, {x, y}, z);

  if (!inplace) {
    dbl::comm::equip_dil_buffer(result, z);
  }
  return result;
}

at::Tensor& AtenIpexCPUDev::dil_add_out(at::Tensor& result, const at::Tensor& self, const at::Tensor& other, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_add_out\n");

  return dil_add_common</*inplace=*/false>(result, self, other, alpha);
}

at::Tensor AtenIpexCPUDev::dil_add(const at::Tensor& self, const at::Tensor& other, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_add\n");

  auto result = dbl::comm::empty_dil_tensor({0}, self.options());
  return dil_add_common</*inplace=*/false>(result, self, other, alpha);
}

at::Tensor & AtenIpexCPUDev::dil_add_(at::Tensor& self, const at::Tensor& other, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_add_\n");

  return dil_add_common</*inplace=*/true>(self, self, other, alpha);
}

dil::tensor reshape_tensor_for_broadcast(dil::tensor& src1, dil::tensor& src2) {
  auto diff_ndims = src1.ndims() - src2.ndims();
  auto right = diff_ndims > 0 ? src2 : src1;
  auto& left = diff_ndims > 0 ? src1 : src2;

  diff_ndims = diff_ndims > 0 ? diff_ndims : -diff_ndims;

  dil::dims reshape_dims;
  for (int i = 0; i < diff_ndims; i++) {
    reshape_dims.push_back(1);
  }
  for (int i = 0; i < right.ndims(); i++) {
    reshape_dims.push_back(right.get_dim(i));
  }
  for (int i = left.ndims() - 1; i >= 0; i--) {
    IPEX_CHECK(reshape_dims[i] == left.get_dim(i) || reshape_dims[i] == 1,
               "dil mul not support the shape for broadcast");
  }
  right.reshape(reshape_dims);
  return right;
}

template<bool inplace>
at::Tensor& dil_mul_common(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other) {
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(other);

  // IPEX_CHECK(self.sizes().equals(other.sizes()),
  //     "dil mul not support broadcast yet");

  dbl::comm::reorder_to_bf16_for_mix_prec(self, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(other, true);

  auto x = dbl::comm::try_gen_dil_tensor(self);
  auto y_ = dbl::comm::try_gen_dil_tensor(other);
  // reorder other to the data type of self
  auto y = dbl::comm::reorder_dil_tensor_to_dtype(y_, x.get_data_type());
  auto z = inplace ? x : dil::tensor();

  auto diff_ndims = x.ndims() - y.ndims();
  if (diff_ndims != 0) {
    auto right = reshape_tensor_for_broadcast(x, y);
    dil::binary::compute(diff_ndims > 0 ? x : y, right, z, dil::algorithm::binary_mul);
  } else {
    dil::binary::compute(x, y, z, dil::algorithm::binary_mul);
  }

  if (!inplace) {
    dbl::comm::equip_dil_buffer(result, z);
  }
  return result;
}

at::Tensor& AtenIpexCPUDev::dil_mul_out(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  DEBUG("AtenIpexCPUDev::dil_mul_out\n");

  return dil_mul_common</*inplace=*/false>(result, self, other);
}

at::Tensor AtenIpexCPUDev::dil_mul(const at::Tensor& self, const at::Tensor& other) {
  DEBUG("AtenIpexCPUDev::dil_mul\n");

  auto result = dbl::comm::empty_dil_tensor({0}, self.options());
  return dil_mul_common</*inplace=*/false>(result, self, other);
}

at::Tensor& AtenIpexCPUDev::dil_mul_(at::Tensor& self, const at::Tensor& other) {
  DEBUG("AtenIpexCPUDev::dil_mul_\n");

  IPEX_CHECK(
    self.ndimension() >= other.ndimension(),
    "The size of input tensors doesn't match the broadcast shape for dil_mul_");
  return dil_mul_common</*inplace=*/true>(self, self, other);
}

void matmul_common(
    const dil::tensor &x,
    const dil::tensor &w,
    const dil::tensor &bias,
    dil::tensor &y,
    at::Scalar beta=1,
    at::Scalar alpha=1,
    const dil::attr_t& attr = dil::attr_t()) {
  DEBUG("AtenIpexCPUDev::matmul_common\n");
  float dst_coeff = alpha.to<float>();
  float sum_coeff = beta.to<float>();
  if (!bias.is_empty()) {
    // DNNL only supports bias in 1xN dims
    // use bias for sum can save tensor memory copy
    if (dst_coeff == 1.0f  && sum_coeff == 1.0f && bias.get_dim(0) == 1) {
      dil::matmul_forward::compute(x, w, bias, y);
      return;
    }
    dil::direct_copy::compute(bias, y);
  }

  dil::matmul_forward::compute(x, w, y, dst_coeff, sum_coeff,
      dil::scale_t(), dil::scale_t(), dil::scale_t(), attr);
}

at::Tensor AtenIpexCPUDev::dil_bmm(const at::Tensor& self, const at::Tensor& mat2) {
  DEBUG("AtenIpexCPUDev::dil_bmm\n");

  auto result = dbl::comm::empty_dil_tensor({0}, self.options());
  return dil_bmm_out(result, self, mat2);
}

at::Tensor& AtenIpexCPUDev::dil_bmm_out(at::Tensor &result, const at::Tensor& batch1, const at::Tensor& batch2) {
  DEBUG("AtenIpexCPUDev::dil_bmm_out\n");
  CHECK_DNNL_OP_PRE_COND(batch1);
  CHECK_DNNL_OP_PRE_COND(batch2);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batch1.dim() == 3 && batch2.dim() == 3);
  dil::dims inferred_size{dil_size(batch1, 0), dil_size(batch1, 1), dil_size(batch2, 2)};

  dbl::comm::reorder_to_bf16_for_mix_prec(batch1);
  dbl::comm::reorder_to_bf16_for_mix_prec(batch2);

  auto x = dbl::comm::try_gen_dil_tensor(batch1);
  auto w = dbl::comm::try_gen_dil_tensor(batch2);
  dil::tensor y;
  matmul_common(x, w, dil::tensor(), y);

  dbl::comm::equip_dil_buffer(result, y);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(inferred_size));
  return result;
}

at::Tensor AtenIpexCPUDev::dil_mm(const at::Tensor& self, const at::Tensor& mat2) {
  DEBUG("AtenIpexCPUDev::dil_mm\n");

  auto result = dbl::comm::empty_dil_tensor({0}, self.options());
  return dil_mm_out(result, self, mat2);
}

at::Tensor& AtenIpexCPUDev::dil_mm_out(at::Tensor& result, const at::Tensor& self, const at::Tensor& mat2) {
  DEBUG("AtenIpexCPUDev::dil_mm_out\n");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.dim() == 2 && mat2.dim() == 2);
  dil::dims inferred_size{dil_size(self, 0), dil_size(mat2, 1)};

  dbl::comm::reorder_to_bf16_for_mix_prec(self);
  dbl::comm::reorder_to_bf16_for_mix_prec(mat2);

  auto x = dbl::comm::try_gen_dil_tensor(self);
  auto w = dbl::comm::try_gen_dil_tensor(mat2);
  dil::tensor y;
  matmul_common(x, w, dil::tensor(), y);

  dbl::comm::equip_dil_buffer(result, y);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(inferred_size));
  return result;
}

template <bool inplace>
at::Tensor& dil_baddbmm_common(
    at::Tensor &result,
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    at::Scalar beta,
    at::Scalar alpha) {
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(batch1);
  CHECK_DNNL_OP_PRE_COND(batch2);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batch1.dim() == 3 && batch2.dim() == 3);
  dil::dims inferred_size{AtenIpexCPUDev::dil_size(batch1, 0), AtenIpexCPUDev::dil_size(batch1, 1), AtenIpexCPUDev::dil_size(batch2, 2)};
  IPEX_CHECK(self.sizes().equals(inferred_size),
      "dil baddbmm not support broadcast yet");

  dbl::comm::reorder_to_bf16_for_mix_prec(self);
  dbl::comm::reorder_to_bf16_for_mix_prec(batch1);
  dbl::comm::reorder_to_bf16_for_mix_prec(batch2);

  auto x = dbl::comm::try_gen_dil_tensor(batch1);
  auto w = dbl::comm::try_gen_dil_tensor(batch2);
  dil::tensor bias;
  if (self.numel() != 0) {
    bias = dbl::comm::try_gen_dil_tensor(self);
    if (bias.ndims() < x.ndims()) {
      auto bias_dims = bias.get_dims();
      bias_dims.insert(bias_dims.begin(), 1);
      bias.reshape(bias_dims);
    }
  }
  auto y = inplace ? dbl::comm::try_gen_dil_tensor(self) : dil::tensor();
  auto attr_ = dil::attr_t::fuse_sum();
  matmul_common(x, w, bias, y, beta, alpha, attr_);

  if (!inplace) {
    dbl::comm::equip_dil_buffer(result, y);
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(inferred_size));
  return result;
}

at::Tensor& AtenIpexCPUDev::dil_baddbmm_out(
    at::Tensor &result,
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    at::Scalar beta,
    at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_baddbmm_out\n");

  return dil_baddbmm_common</*inplace=*/false>(result, self, batch1, batch2, beta, alpha);
}

at::Tensor AtenIpexCPUDev::dil_baddbmm(const at::Tensor& self, const at::Tensor& batch1, const at::Tensor & batch2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_baddbmm\n");

  auto result = dbl::comm::empty_dil_tensor({0}, self.options());
  return dil_baddbmm_common</*inplace=*/false>(result, self, batch1, batch2, beta, alpha);
}

at::Tensor& AtenIpexCPUDev::dil_baddbmm_(at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_baddbmm_\n");

  return dil_baddbmm_out(self, self, batch1, batch2, beta, alpha);
}

template<bool inplace>
at::Tensor& dil_addmm_common(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    at::Scalar beta,
    at::Scalar alpha) {
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(mat1);
  CHECK_DNNL_OP_PRE_COND(mat2);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat1.dim() == 2 && mat2.dim() == 2);
  dil::dims inferred_size{AtenIpexCPUDev::dil_size(mat1, 0), AtenIpexCPUDev::dil_size(mat2, 1)};
  IPEX_CHECK(self.sizes().equals(inferred_size),
      "dil addmm not support broadcast yet");

  dbl::comm::reorder_to_bf16_for_mix_prec(self);
  dbl::comm::reorder_to_bf16_for_mix_prec(mat1);
  dbl::comm::reorder_to_bf16_for_mix_prec(mat2);

  auto x = dbl::comm::try_gen_dil_tensor(mat1);
  auto w = dbl::comm::try_gen_dil_tensor(mat2);
  dil::tensor bias;
  if (self.numel() != 0) {
    bias = dbl::comm::try_gen_dil_tensor(self);
    if (bias.ndims() < x.ndims()) {
      auto bias_dims = bias.get_dims();
      bias_dims.insert(bias_dims.begin(), 1);
      bias.reshape(bias_dims);
    }
  }
  auto y = inplace ? dbl::comm::try_gen_dil_tensor(self) : dil::tensor();
  auto attr_ = dil::attr_t::fuse_sum();
  matmul_common(x, w, bias, y, beta, alpha, attr_);

  if (!inplace) {
    dbl::comm::equip_dil_buffer(result, y);
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(inferred_size));
  return result;
}

at::Tensor& AtenIpexCPUDev::dil_addmm_out(at::Tensor& result, const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_addmm_out\n");

  return dil_addmm_common</*inplace=*/false>(result, self, mat1, mat2, beta, alpha);
}

at::Tensor AtenIpexCPUDev::dil_addmm(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor & mat2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_addmm\n");

  auto result = dbl::comm::empty_dil_tensor({0}, self.options());
  return dil_addmm_common</*inplace=*/false>(result, self, mat1, mat2, beta, alpha);
}

at::Tensor& AtenIpexCPUDev::dil_addmm_(at::Tensor& self, const at::Tensor& mat1, const at::Tensor & mat2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_addmm_\n");

  return dil_addmm_common</*inplace=*/false>(self, self, mat1, mat2, beta, alpha);
}

template<bool inplace>
at::Tensor& dil_addbmm_common(
    at::Tensor& result,
    const at::Tensor &self,
    const at::Tensor &batch1,
    const at::Tensor &batch2,
    at::Scalar beta,
    at::Scalar alpha) {
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(batch1);
  CHECK_DNNL_OP_PRE_COND(batch2);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batch1.dim() == 3 && batch2.dim() == 3);
  dil::dims inferred_size{AtenIpexCPUDev::dil_size(batch1, 1), AtenIpexCPUDev::dil_size(batch2, 2)};
  IPEX_CHECK(self.sizes().equals(inferred_size),
      "dil addbmm not support broadcast yet");

  dbl::comm::reorder_to_bf16_for_mix_prec(self);
  dbl::comm::reorder_to_bf16_for_mix_prec(batch1);
  dbl::comm::reorder_to_bf16_for_mix_prec(batch2);

  // addbmm(batch1*batch2) [b,n,m] * [b,m,p] = [n,p] can be treated as:
  // [n, b*m] * [b*m, p] = [n, p]
  // For batch1: reorder from [b, n, m] to [n, b, m], reshape to [n, b*m]
  // For batch2: reshape from [b, m, p] to [b*m, p]
  auto x = dbl::comm::try_gen_dil_tensor(batch1);
  auto w = dbl::comm::try_gen_dil_tensor(batch2);

  auto x_ = x;
  if (x.get_dim(0) > 1) {
    x_ = x.transpose(0, 1);
  }
  x_ = x_.reshape({x.get_dim(1), x.get_dim(0) * x.get_dim(2)});
  auto w_ = w.reshape({w.get_dim(0) * w.get_dim(1), w.get_dim(2)});
  auto y = inplace ? dbl::comm::try_gen_dil_tensor(self) : dil::tensor();
  auto attr_ = dil::attr_t::fuse_sum();

  dil::tensor bias;
  if (self.numel() != 0) {
    bias = dbl::comm::try_gen_dil_tensor(self);
    if (bias.ndims() < x_.ndims()) {
      auto bias_dims = bias.get_dims();
      bias_dims.insert(bias_dims.begin(), 1);
      bias.reshape(bias_dims);
    }
  }
  matmul_common(x_, w_, bias, y, beta, alpha, attr_);

  if (!inplace) {
    dbl::comm::equip_dil_buffer(result, y);
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(inferred_size));
  return result;
}

at::Tensor& AtenIpexCPUDev::dil_addbmm_out(at::Tensor& result, const at::Tensor &self, const at::Tensor &batch1, const at::Tensor &batch2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_addbmm_out\n");

  return dil_addbmm_common</*inplace=*/false>(result, self, batch1, batch2, beta, alpha);
}

at::Tensor AtenIpexCPUDev::dil_addbmm(const at::Tensor &self, const at::Tensor &batch1, const at::Tensor &batch2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_addbmm\n");

  auto result = dbl::comm::empty_dil_tensor({0}, self.options());
  return dil_addbmm_common</*inplace=*/false>(result, self, batch1, batch2, beta, alpha);
}

at::Tensor& AtenIpexCPUDev::dil_addbmm_(at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2, at::Scalar beta, at::Scalar alpha) {
  DEBUG("AtenIpexCPUDev::dil_addbmm_\n");

  return dil_addbmm_common</*inplace=*/true>(self, self, batch1, batch2, beta, alpha);
}

at::Tensor AtenIpexCPUDev::dil_linear(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const dil::attr_t& attr) {
  DEBUG("AtenIpexCPUDev::dil_linear\n");
  CHECK_DNNL_OP_PRE_COND(self);
  CHECK_DNNL_OP_PRE_COND(weight);
  IPEX_CHECK(self.dim() >= 2,
      "dil_linear: input needs to has dim at least 2, input dim ", self.dim());
  IPEX_CHECK(attr.get_post_ops().len() == 0 || 
                              attr.get_post_ops().kind(0) == dnnl::primitive::kind::eltwise, 
                              "dil linear only fuse with eltwise now");

  std::vector<float> output_scale = {};
  bool quantized = false;
  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<std::vector<float>> scales;
    std::tie(scales, quantized) = dbl::comm::get_int8_scales({self}, /*  uint8_used for output*/false);
    //quantized = false;
    if (quantized) {
      output_scale.push_back(scales[1][0]);
      dbl::comm::reorder_to_int8_for_mix_prec(self, scales[0]);
      dbl::comm::reorder_to_int8_for_mix_prec(weight, {});
    } else {
      dbl::comm::reorder_to_dtype(self, at::kFloat);
      dbl::comm::reorder_to_dtype(weight, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(self);
    dbl::comm::reorder_to_bf16_for_mix_prec(weight, true);
  }

  // reshape first if input dim is greater than 2 and the reshape will cost a memory copy.
  auto self_reshaped = self.dim() > 2 ? dil_reshape(self, {-1, dil_size(self, self.dim() - 1)}) : self;
  const dil::tensor x = dbl::comm::try_gen_dil_tensor(self_reshaped);
  if (!check_train()) {
    dbl::linear::prepack_linear_weights(self_reshaped, x, weight);
  }
  const dil::tensor w = dbl::comm::try_gen_dil_tensor(weight);

  c10::optional<dil::tensor> b{c10::nullopt};
  if (bias.defined()) {
    if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
      if (quantized) {
        auto src = dbl::comm::try_gen_dil_storage(bias);
        auto src_type = src.get_data_type();
        if (src_type != dil::data_type::s32) {
          auto dst_desc = src.get_desc().to_type(dil::data_type::s32);
          auto bias_scales = w.get_scale();
          for (auto &scale : bias_scales) { scale *= x.get_scale()[0];  }
          dbl::comm::reorder_to_desc(bias, dst_desc, bias_scales);
        }
      } else {
        dbl::comm::reorder_to_dtype(bias, at::kFloat);
      }
    } else {
      dbl::comm::reorder_to_bf16_for_mix_prec(bias, true);
    }
    b = dbl::comm::try_gen_dil_tensor(bias);
  }

  dil::tensor y = dbl::linear::linear_impl(x, w, b, output_scale, attr);

  if (self.dim() > 2) {
    auto input_size = self.sizes();
    std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
    output_size.push_back(dil_size(weight, 0));
    y.reshape(output_size);
  }

  auto aten_output = dbl::comm::gen_aten_tensor_by(std::move(y));

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    auto op_name = attr.get_post_ops().len() == 0 ? "Linear" : "LinearFuseEltwise";
    insert_or_updata_observer({self}, {aten_output}, op_name);
  }

  return aten_output;
}

at::Tensor AtenIpexCPUDev::dil_linear_backward_input(
    at::IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight){
  DEBUG("AtenIpexCPUDev::dil_linear_backward_input\n");

  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(weight);
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dbl::comm::reorder_to_bf16_for_mix_prec(weight, true);

  auto grad_output_reshaped = grad_output.dim() > 2 ?
    dil_reshape(grad_output, {-1, dil_size(grad_output, grad_output.dim() - 1)}) : grad_output;
  dil::tensor grady = dbl::comm::try_gen_dil_tensor(grad_output_reshaped);
  const dil::tensor w = dbl::comm::try_gen_dil_tensor(weight);

  std::vector<int64_t> input_reshaped_size;
  input_reshaped_size.push_back(dil_size(grad_output_reshaped, 0));
  input_reshaped_size.push_back(dil_size(weight, 1));

  dil::tensor gradx;
  dil::inner_product_backward_data::compute(
    grady, w, {input_reshaped_size.begin(), input_reshaped_size.end()}, gradx);

  if (input_size.size() > 2) {
    return dbl::comm::gen_aten_tensor_by(std::move(gradx)).reshape(input_size);
  }
  return dbl::comm::gen_aten_tensor_by(std::move(gradx));
}

std::tuple<at::Tensor, at::Tensor> AtenIpexCPUDev::dil_linear_backward_weights(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight, bool bias_defined) {
  DEBUG("AtenIpexCPUDev::dil_linear_backward_weights\n");

  CHECK_DNNL_OP_PRE_COND(input);
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(weight);
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output);
  dbl::comm::reorder_to_bf16_for_mix_prec(input);
  dbl::comm::reorder_to_bf16_for_mix_prec(weight, true);

  auto grad_output_reshaped = grad_output.dim() > 2 ?
    dil_reshape(grad_output, {-1, dil_size(grad_output, grad_output.dim() - 1)}) : grad_output;
  auto input_reshaped = input.dim() > 2 ? dil_reshape(input, {-1, dil_size(input, input.dim() - 1)}) : input;

  dil::tensor grady = dbl::comm::try_gen_dil_tensor(grad_output_reshaped);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(input_reshaped);
  dil::tensor w = dbl::comm::try_gen_dil_tensor(weight);
  auto diff_weight_type = w.get_data_type();

  dil::tensor gradw, gradb;
  if (bias_defined) {
    dil::inner_product_backward_weights::compute(x, grady, gradw, gradb, diff_weight_type);
    return std::tuple<at::Tensor, at::Tensor>{
    dbl::comm::gen_aten_tensor_by(std::move(gradw)),
    dbl::comm::gen_aten_tensor_by(std::move(gradb))};
  } else {
    dil::inner_product_backward_weights::compute(x, grady, gradw, diff_weight_type);
    return std::tuple<at::Tensor, at::Tensor>{
    dbl::comm::gen_aten_tensor_by(std::move(gradw)),
    at::Tensor()};
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenIpexCPUDev::dil_linear_backward(
    const at::Tensor& input, const at::Tensor& grad_output,
    const at::Tensor& weight, std::array<bool,3> output_mask) {
  DEBUG("AtenIpexCPUDev::dil_linear_backward\n");

  at::Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = dil_linear_backward_input(input.sizes(), grad_output, weight);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = dil_linear_backward_weights(grad_output, input, weight, output_mask[2]);
  }
  return std::tuple<at::Tensor, at::Tensor, at::Tensor>{grad_input, grad_weight, grad_bias};
}

std::tuple<at::Tensor, at::Tensor> _dil_dropout(
    const at::Tensor& self,
    double ratio) {
  IPEX_CHECK(
      ratio >= 0 && ratio < 1 && self.numel() != 0,
      "dropout probability has to be between 0 and 1, but got ",
      ratio);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::tensor mask;
  dil::tensor y;
  dil::dropout_forward::compute(x, ratio, y, mask);
  return std::tuple<at::Tensor, at::Tensor>{
      dbl::comm::gen_aten_tensor_by(std::move(y)),
      dbl::comm::gen_aten_tensor_by(std::move(mask))};
}

at::Tensor AtenIpexCPUDev::dil_dropout(const at::Tensor& self, double ratio, bool train) {
  DEBUG("AtenIpexCPUDev::dil_dropout\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(self, true);

  return std::get<0>(_dil_dropout(self, ratio));
}

at::Tensor AtenIpexCPUDev::dil_dropout_backward(
    const at::Tensor& grady,
    const at::Tensor& mask,
    double ratio) {
  DEBUG("AtenIpexCPUDev::dil_dropout_backward\n");
  CHECK_DNNL_OP_PRE_COND(grady);
  if (ratio == 0 || grady.numel() == 0) {
    return grady;
  }

  dbl::comm::reorder_to_bf16_for_mix_prec(grady, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(mask, true);

  dil::tensor dY = dbl::comm::try_gen_dil_tensor(grady);
  dil::tensor mask_dil = dbl::comm::try_gen_dil_tensor(mask);
  dil::tensor dX;
  dil::dropout_backward::compute(mask_dil, dY, dX);
  return dbl::comm::gen_aten_tensor_by(std::move(dX));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenIpexCPUDev::dil_native_batch_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  DEBUG("AtenIpexCPUDev::dil_native_batch_norm\n");
  CHECK_DNNL_OP_PRE_COND(input);
  CHECK_DNNL_OP_PRE_COND(weight);
  IPEX_CHECK(input.dim() == 4 || input.dim() == 5,
             "mkldnn_batch_norm: currently mkldnn only support 2d and 3d batchnorm");
  IPEX_CHECK(weight.defined() && bias.defined(),
             "mkldnn_batch_norm: currently mkldnn only support affine model");

  if (check_auto_mix_int8_fp32()) {
    IPEX_CHECK(!train, "mkldnn_bacth_norm: mkldnn only support inference model for int8");
  }
  std::vector<float> input_scales = {};
  std::vector<float> output_scales = {};
  bool quantized = false;
  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<std::vector<float>> scales;
    std::tie(scales, quantized) = dbl::comm::get_int8_scales({input}, /*  uint8_used for output*/false);
    //quantized = false;
    if (quantized) {
      input_scales = scales[0];
      output_scales = scales[1];
      dbl::comm::reorder_to_int8_for_mix_prec(input, input_scales);
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input, true);
  }

  dil::tensor x = dbl::comm::try_gen_dil_tensor(input);
  const dil::tensor w = dbl::comm::try_gen_dil_tensor(weight);
  const dil::tensor b = dbl::comm::try_gen_dil_tensor(bias);
  bool use_running_stat = (running_mean.defined() && running_var.defined());
  dil::tensor y;
  if (train) {
    dil::tensor saved_mean;
    dil::tensor saved_var;
    dil::batch_normalization_forward_training::compute(
        x, w, b, y, saved_mean, saved_var, momentum, eps);
    if (use_running_stat) {
      auto len = x.get_nelems() / w.get_nelems(); // n*h*w
      dil::tensor m = dbl::comm::try_gen_dil_tensor(running_mean);
      dil::tensor v = dbl::comm::try_gen_dil_tensor(running_var);
      const std::vector<float> scales_mean{1 - float(momentum), float(momentum)};
      const std::vector<float> scales_var{1 - float(momentum), float(momentum) * len / (len - 1)};
      dil::sum::compute(scales_mean, {m, saved_mean}, m);
      dil::sum::compute(scales_var, {v, saved_var}, v);
    }
    return std::make_tuple(
        dbl::comm::gen_aten_tensor_by(std::move(y)),
        dbl::comm::gen_aten_tensor_by(std::move(saved_mean)),
        dbl::comm::gen_aten_tensor_by(std::move(saved_var)));
  } else {
    if (use_running_stat) {
      dil::tensor m = dbl::comm::try_gen_dil_tensor(running_mean);
      dil::tensor v = dbl::comm::try_gen_dil_tensor(running_var);
      dil::batch_normalization_forward_inference::compute(
          x, m, v, w, b, y, eps, input_scales, output_scales);
    } else {
      dil::batch_normalization_forward_inference::compute(
          x, w, b, y, eps, input_scales, output_scales);
    }

    auto aten_output = dbl::comm::gen_aten_tensor_by(std::move(y));

    if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
      insert_or_updata_observer({input}, {aten_output}, "BatchNorm");
    }

    return std::make_tuple(aten_output, at::Tensor(), at::Tensor());
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenIpexCPUDev::dil_native_batch_norm_backward(const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    bool train,
    double eps,
    std::array<bool,3> grad_input_mask) {
  DEBUG("AtenIpexCPUDev::dil_native_batch_norm_backward\n");
  CHECK_DNNL_OP_PRE_COND(input);
  CHECK_DNNL_OP_PRE_COND(weight);

  IPEX_CHECK(train, "mkldnn_batch_norm_backward: currently mkldnn only support train model");
  auto grad_output_contiguous = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output_contiguous, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(input, true);

  dil::tensor grady = dbl::comm::try_gen_dil_tensor(grad_output_contiguous);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(input);
  dil::tensor w = dbl::comm::try_gen_dil_tensor(weight);
  dil::tensor m = dbl::comm::try_gen_dil_tensor(save_mean);
  dil::tensor v = dbl::comm::try_gen_dil_tensor(save_invstd);

  dil::tensor gradx, gradw, gradb;
  dil::batch_normalization_backward::compute(
      x, m, v, grady, w, gradx, gradw, gradb, eps);

  return std::make_tuple(
      dbl::comm::gen_aten_tensor_by(std::move(gradx)),
      dbl::comm::gen_aten_tensor_by(std::move(gradw)),
      dbl::comm::gen_aten_tensor_by(std::move(gradb)));
}

at::Tensor AtenIpexCPUDev::dil_max_pooling(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  DEBUG("AtenIpexCPUDev::dil_max_pooling\n");
  CHECK_DNNL_OP_PRE_COND(input);
  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<std::vector<float>> scales;
    bool quantized;
    std::tie(scales, quantized) = dbl::comm::get_int8_scales({input}, /*  uint8_used for output*/false);
    //quantized = false;
    if (quantized) {
      dbl::comm::reorder_to_int8_for_mix_prec(input, scales[0]);
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input, true);
  }

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer({input}, {input}, "MaxPooling");
  }
  return dbl::pool::_dil_pooling(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      dil::algorithm::pooling_max);
}

at::Tensor AtenIpexCPUDev::dil_avg_pool2d(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  DEBUG("AtenIpexCPUDev::dil_avg_pool2d\n");
  CHECK_DNNL_OP_PRE_COND(input);
  IPEX_CHECK(!divisor_override.has_value(),
           "dil_avg_pooling operator does not support divisor");

  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<std::vector<float>> scales;
    bool quantized;
    std::tie(scales, quantized) = dbl::comm::get_int8_scales({input}, /*  uint8_used for output*/false);
    //quantized = false;
    if (quantized) {
      dbl::comm::reorder_to_int8_for_mix_prec(input, scales[0]);
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input, true);
  }

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer({input}, {input}, "AvgPool2d");
  }

  return dbl::pool::_dil_pooling(
      input.is_contiguous() ? input : input.contiguous(),
      kernel_size,
      stride,
      padding,
      /* dilation*/ std::vector<int64_t> {1, 1},
      ceil_mode,
      count_include_pad ? dil::algorithm::pooling_avg_include_padding
                        : dil::algorithm::pooling_avg_exclude_padding);
}

at::Tensor AtenIpexCPUDev::dil_avg_pool3d(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  DEBUG("AtenIpexCPUDev::dil_avg_pool3d\n");
  CHECK_DNNL_OP_PRE_COND(input);
  IPEX_CHECK(!divisor_override.has_value(),
           "dil_avg_pooling operator does not support divisor");

  dbl::comm::reorder_to_bf16_for_mix_prec(input, true);

  return dbl::pool::_dil_pooling(
      input.is_contiguous() ? input : input.contiguous(),
      kernel_size,
      stride,
      padding,
      /* dilation*/ std::vector<int64_t> {1, 1, 1},
      ceil_mode,
      count_include_pad ? dil::algorithm::pooling_avg_include_padding
                        : dil::algorithm::pooling_avg_exclude_padding);
}

at::Tensor AtenIpexCPUDev::dil_adaptive_avg_pool2d(
    const at::Tensor& input,
    at::IntArrayRef output_size) {
  DEBUG("AtenIpexCPUDev::dil_adaptive_avg_pool2d\n");
  CHECK_DNNL_OP_PRE_COND(input);

  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<std::vector<float>> scales;
    bool quantized;
    std::tie(scales, quantized) = dbl::comm::get_int8_scales({input}, /*  uint8_used for output*/false);
    //quantized = false;
    if (quantized) {
      dbl::comm::reorder_to_int8_for_mix_prec(input, scales[0]);
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input, true);
  }

  auto output_size_vec =
      dbl::comm::expand_param_if_needed(output_size, "output_size", input.dim() - 2);
  std::vector<int64_t> kernel_size(input.dim() - 2);
  for (int64_t i = 2; i < input.dim(); ++i) {
    auto s1 = dil_size(input, i);
    auto s2 = output_size_vec[i - 2];
    IPEX_CHECK(s2 != 0, "output size can not be zero");
    IPEX_CHECK(
        s1 % s2 == 0,
        "input size is not divisible by the output size is not supported yet");
    kernel_size[i - 2] = s1 / s2;
  }
  std::vector<int64_t> padding{0, 0};
  std::vector<int64_t> dilation{1, 1};

  if (input.dim() == 5) {
    padding.push_back(0);
    dilation.push_back(1);
  }

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer({input}, {input}, "AdaptiveAvgPool2d");
  }
  return dbl::pool::_dil_pooling(
      input,
      kernel_size,
      /*stride*/ kernel_size,
      /*padding*/ padding,
      /*dilation*/ dilation,
      /*ceil_mode*/ false,
      /*algo*/ dil::algorithm::pooling_avg);
}

at::Tensor AtenIpexCPUDev::dil_max_pooling_backward(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  DEBUG("AtenIpexCPUDev::dil_max_pooling_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(output);
  CHECK_DNNL_OP_PRE_COND(input);

  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(output, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(input, true);

  return dbl::pool::_dil_pooling_backward(
      grad_output.is_contiguous() ? grad_output : grad_output.contiguous(),
      output.is_contiguous() ? output : output.contiguous(),
      input.is_contiguous() ? input : input.contiguous(),
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      dil::algorithm::pooling_max);
}

at::Tensor AtenIpexCPUDev::dil_avg_pool2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  DEBUG("AtenIpexCPUDev::dil_avg_pool2d_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(input);

  auto grad_output_contiguous = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output_contiguous, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(input, true);

  return dbl::pool::_dil_pooling_backward(
      grad_output_contiguous,
      grad_output_contiguous,
      input.is_contiguous() ? input : input.contiguous(),
      kernel_size,
      stride,
      padding,
      /* dilation */ std::vector<int64_t>{1, 1},
      ceil_mode,
      count_include_pad ? dil::algorithm::pooling_avg_include_padding
                        : dil::algorithm::pooling_avg_exclude_padding);
}

at::Tensor AtenIpexCPUDev::dil_avg_pool3d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  DEBUG("AtenIpexCPUDev::dil_avg_pool3d_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(input);

  auto grad_output_contiguous = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output_contiguous, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(input, true);

  std::vector<int64_t> dilation{1, 1};
  return dbl::pool::_dil_pooling_backward(
      grad_output_contiguous,
      grad_output_contiguous,
      input.is_contiguous() ? input : input.contiguous(),
      kernel_size,
      stride,
      padding,
      /* dilation */ std::vector<int64_t>{1, 1, 1},
      ceil_mode,
      count_include_pad ? dil::algorithm::pooling_avg_include_padding
                        : dil::algorithm::pooling_avg_exclude_padding);
}

at::Tensor AtenIpexCPUDev::dil_adaptive_avg_pool2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input) {
  DEBUG("AtenIpexCPUDev::dil_adaptive_avg_pool2d_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(input);

  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(input, true);

  auto output_size_vec = grad_output.sizes();
  std::vector<int64_t> kernel_size(input.dim() - 2);
  for (size_t i = 2; i < input.dim(); ++i) {
    auto s1 = dil_size(input, i);
    auto s2 = output_size_vec[i];
    IPEX_CHECK(s2 != 0, "output size can not be zero");
    IPEX_CHECK(
        s1 % s2 == 0,
        "input size is not divisible by the output size is not supported yet");
    kernel_size[i - 2] = s1 / s2;
  }
  std::vector<int64_t> padding{0, 0};
  std::vector<int64_t> dilation{1, 1};

  if (input.dim() == 5) {
    padding.push_back(0);
    dilation.push_back(1);
  }

  return dbl::pool::_dil_pooling_backward(
      grad_output,
      grad_output,
      input.is_contiguous() ? input : input.contiguous(),
      kernel_size,
      /*stride*/ kernel_size,
      /*padding*/ padding,
      /*dilation*/ dilation,
      false,
      /*algo*/ dil::algorithm::pooling_avg);
}

at::Tensor AtenIpexCPUDev::dil_relu(const at::Tensor& input) {
  DEBUG("AtenIpexCPUDev::dil_relu\n");
  CHECK_DNNL_OP_PRE_COND(input);
  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<std::vector<float>> scales;
    bool quantized;
    std::tie(scales, quantized)= dbl::comm::get_int8_scales({input}, /*  uint8_used for output*/true);
    //quantized = false;
    if (quantized) {
      dbl::comm::reorder_to_int8_for_mix_prec(input, scales[0]);
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input, true);
  }

  const dil::tensor& x = dbl::comm::try_gen_dil_tensor(input);
  dil::tensor y;
  dil::eltwise_forward::compute(
      x, y, dil::algorithm::eltwise_relu, dil::prop_kind::forward_training, /*alpha*/ 0.0);

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer({input}, {input}, "Relu");
  }

  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

at::Tensor& AtenIpexCPUDev::dil_relu_(at::Tensor& input) {
  DEBUG("AtenIpexCPUDev::dil_relu_\n");
  CHECK_DNNL_OP_PRE_COND(input);

  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    std::vector<std::vector<float>> scales;
    bool quantized;
    std::tie(scales, quantized) = dbl::comm::get_int8_scales({input}, /*   uint8_used for output*/true);
    //quantized = false;
    if (quantized) {
      dbl::comm::reorder_to_int8_for_mix_prec(input, scales[0]);
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input, true);
  }

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer({input}, {input}, "Relu_");
  }

  auto dil_self = dbl::comm::try_gen_dil_tensor(input);
  dil::eltwise_forward::compute(
    dil_self,
    dil_self,
    dil::algorithm::eltwise_relu,
    dil::prop_kind::forward_training,
    /*alpha*/ 0.0);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dil_self.is_public_format() || check_tensor_own_whole_storage(input));
  dbl::comm::sync_shape_from_dil_to_aten(input, dil_self);
  return input;
}

at::Tensor AtenIpexCPUDev::dil_threshold_backward(const at::Tensor& grad_output, const at::Tensor& input, at::Scalar threshold) {
  DEBUG("AtenIpexCPUDev::dil_threshold_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(input);

  auto grad_output_contiguous = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output_contiguous, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(input, true);

  // TODO: support bounded relu. `threshold` is ignored for now
  dil::tensor x = dbl::comm::try_gen_dil_tensor(input);
  dil::tensor grady = dbl::comm::try_gen_dil_tensor(grad_output_contiguous);
  dil::tensor gradx;
  dil::eltwise_backward::compute(x, grady, gradx,
      dil::algorithm::eltwise_relu, /*alpha*/ 0.0);
  return dbl::comm::gen_aten_tensor_by(std::move(gradx));
}

at::Tensor AtenIpexCPUDev::dil__softmax(
    const at::Tensor& self,
    const int64_t dim,
    bool half_to_float) {
  DEBUG("AtenIpexCPUDev::dil__softmax\n");
  CHECK_DNNL_OP_PRE_COND(self);
  AT_ASSERTM(
      !half_to_float,
      "softmax with half to float conversion is not supported on Mkldnn");

  dbl::comm::reorder_to_bf16_for_mix_prec(self, true);

  const int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::tensor y;
  dil::softmax_forward::compute(x, y, wrapped_dim);
  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

at::Tensor AtenIpexCPUDev::dil__softmax_backward_data(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    const at::Tensor& self) {
  DEBUG("AtenIpexCPUDev::dil__softmax_backward_data\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(output);
  CHECK_DNNL_OP_PRE_COND(self);

  auto grad_output_contiguous = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output_contiguous, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(output, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(self, true);

  const int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
  dil::tensor y = dbl::comm::try_gen_dil_tensor(output);
  dil::tensor grady = dbl::comm::try_gen_dil_tensor(grad_output_contiguous);
  dil::tensor gradx;
  dil::softmax_backward::compute(y, grady, gradx, wrapped_dim);
  return dbl::comm::gen_aten_tensor_by(std::move(gradx));
}

at::Tensor AtenIpexCPUDev::dil__log_softmax(
    const at::Tensor& self,
    const int64_t dim,
    bool half_to_float) {
  DEBUG("AtenIpexCPUDev::dil__log_softmax\n");
  CHECK_DNNL_OP_PRE_COND(self);
  AT_ASSERTM(
      !half_to_float,
      "softmax with half to float conversion is not supported on Mkldnn");

  dbl::comm::reorder_to_bf16_for_mix_prec(self, true);

  const int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::tensor y;
  dil::logsoftmax_forward::compute(x, y, wrapped_dim);
  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

at::Tensor AtenIpexCPUDev::dil__log_softmax_backward_data(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    const at::Tensor& self) {
  DEBUG("AtenIpexCPUDev::dil__log_softmax_backward_data\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(output);
  CHECK_DNNL_OP_PRE_COND(self);

  auto grad_output_contiguous = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output_contiguous, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(output, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(self, true);

  const int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
  dil::tensor y = dbl::comm::try_gen_dil_tensor(output);
  dil::tensor grady = dbl::comm::try_gen_dil_tensor(grad_output_contiguous);
  dil::tensor gradx;
  dil::logsoftmax_backward::compute(y, grady, gradx, wrapped_dim);
  return dbl::comm::gen_aten_tensor_by(std::move(gradx));
}

at::Tensor AtenIpexCPUDev::dil_sigmoid(const at::Tensor& self) {
  DEBUG("AtenIpexCPUDev::dil_sigmoid\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(self, true);

  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::tensor y;
  dil::eltwise_forward::compute(
      x, y, dil::algorithm::eltwise_logistic_use_dst_for_bwd, dil::prop_kind::forward);
  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

at::Tensor& AtenIpexCPUDev::dil_sigmoid_(at::Tensor& self) {
  DEBUG("AtenIpexCPUDev::dil_sigmoid_\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(self, true);

  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::eltwise_forward::compute(
      x, x, dil::algorithm::eltwise_logistic_use_dst_for_bwd, dil::prop_kind::forward);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(x.is_public_format() || check_tensor_own_whole_storage(self));
  dbl::comm::sync_shape_from_dil_to_aten(self, x);
  return self;
}

at::Tensor AtenIpexCPUDev::dil_sigmoid_backward(
    const at::Tensor& grad_output,
    const at::Tensor& output) {
  DEBUG("AtenIpexCPUDev::dil_sigmoid_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(output);

  auto grad_output_contiguous = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output_contiguous, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(output, true);

  dil::tensor y = dbl::comm::try_gen_dil_tensor(output);
  dil::tensor gy = dbl::comm::try_gen_dil_tensor(grad_output_contiguous);
  dil::tensor gx;
  dil::eltwise_backward::compute(y, gy, gx,
      dil::algorithm::eltwise_logistic_use_dst_for_bwd);
  return dbl::comm::gen_aten_tensor_by(std::move(gx));
}

at::Tensor AtenIpexCPUDev::dil_tanh(const at::Tensor& self) {
  DEBUG("AtenIpexCPUDev::dil_tanh\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(self, true);

  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::tensor y;
  dil::eltwise_forward::compute(
      x, y, dil::algorithm::eltwise_tanh_use_dst_for_bwd, dil::prop_kind::forward);
  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

at::Tensor& AtenIpexCPUDev::dil_tanh_(at::Tensor& self) {
  DEBUG("AtenIpexCPUDev::dil_tanh_\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(self, true);

  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::eltwise_forward::compute(
      x, x, dil::algorithm::eltwise_tanh_use_dst_for_bwd, dil::prop_kind::forward);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(x.is_public_format() || check_tensor_own_whole_storage(self));
  dbl::comm::sync_shape_from_dil_to_aten(self, x);
  return self;
}

at::Tensor AtenIpexCPUDev::dil_tanh_backward(
    const at::Tensor& grad_output,
    const at::Tensor& output) {
  DEBUG("AtenIpexCPUDev::dil_tanh_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(output);

  auto grad_output_contiguous = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output_contiguous, true);
  dbl::comm::reorder_to_bf16_for_mix_prec(output, true);

  dil::tensor y = dbl::comm::try_gen_dil_tensor(output);
  dil::tensor gy = dbl::comm::try_gen_dil_tensor(grad_output_contiguous);
  dil::tensor gx;
  dil::eltwise_backward::compute(y, gy, gx,
      dil::algorithm::eltwise_tanh_use_dst_for_bwd);
  return dbl::comm::gen_aten_tensor_by(std::move(gx));
}

at::Tensor AtenIpexCPUDev::dil_reshape(const at::Tensor& self, at::IntArrayRef size) {
  DEBUG("AtenIpexCPUDev::dil_reshape\n");
  CHECK_DNNL_OP_PRE_COND(self);
  bool int8_enabled = check_auto_mix_int8_fp32() && check_tensor_own_whole_storage(self);
  // if int8 path enabled and self own whole storage, not need to
  // reorder to fp32 dtype, i.e, direct get dil tensor(fp32, int8).
  if (check_auto_mix_bf16_fp32()) {
    dbl::comm::reorder_to_bf16_for_mix_prec(self, true);
  } else if (!int8_enabled) {
    dbl::comm::reorder_to_dtype(self, at::kFloat);
  }

  auto inferred_size = at::infer_size(size, self.numel());
  if (self.sizes() == inferred_size) {
    return self;
  }
  const dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  dil::tensor y{x};
  y.reshape(inferred_size);
  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

int64_t AtenIpexCPUDev::dil_size(const at::Tensor & self, int64_t dim) {
  DEBUG("AtenIpexCPUDev::dil_size\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dim = at::maybe_wrap_dim(dim, self.dim(), false);
  return self.sizes()[dim];
}

at::Tensor AtenIpexCPUDev::dil_clone(const at::Tensor& self, c10::optional<c10::MemoryFormat> optional_memory_format) {
  DEBUG("AtenIpexCPUDev::dil_clone\n");
  CHECK_DNNL_OP_PRE_COND(self);

  auto memory_format =
      optional_memory_format.value_or(at::MemoryFormat::Preserve);

  if (memory_format == at::MemoryFormat::Preserve) {
    if (!self.is_non_overlapping_and_dense()) {
      memory_format = self.suggest_memory_format();
    }
  }

  IPEX_CHECK(
      memory_format == at::MemoryFormat::Preserve ||
          memory_format == at::MemoryFormat::Contiguous,
      "dil_clone only support Preserve of Contiguous");

  auto src = dbl::comm::try_gen_dil_tensor(self);
  auto dst_desc =
      memory_format == at::MemoryFormat::Preserve
          ? src.get_desc()
          : src.get_desc().to_default_format();
  dil::tensor dst{dst_desc};
  src.reorder_to(dst);

  if (src.has_scale()) {
    dst.set_scale(src.get_scale());
  }
  if (src.has_zero_point()) {
    dst.set_zero_point(src.get_zero_point());
  }
  if (src.has_workspace()) {
    dst.copy_workspace(src);
  }
  return dbl::comm::gen_aten_tensor_by(std::move(dst));
}

inline void check_cat_no_zero_dim(at::TensorList tensors) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto& t = tensors[i];
    IPEX_CHECK(t.dim() > 0,
      "zero-dimensional tensor (at position ", i, ") cannot be concatenated");
  }
}

at::Tensor& AtenIpexCPUDev::dil_cat_out(at::Tensor& result, at::TensorList tensors, int64_t dim) {
  DEBUG("AtenIpexCPUDev::dil_cat_out\n");
  CHECK_DNNL_OP_PRE_COND(result);

  dbl::comm::reorder_to_bf16_for_mix_prec(result, true);

  check_cat_no_zero_dim(tensors);
  dim = at::legacy_cat_wrap_dim(dim, tensors);
  std::vector<dil::tensor> x;
  for (auto i =0; i< tensors.size(); i++) {
    IPEX_CHECK(!(tensors[i].dim() == 1 && tensors[i].sizes()[0] == 0),
      "Currently Mkldnn cat operators do not support empty tensor.");

    dbl::comm::reorder_to_bf16_for_mix_prec(tensors[i], true);

    x.push_back(dbl::comm::try_gen_dil_tensor(tensors[i]));
  }
  dil::tensor y = dbl::comm::try_gen_dil_tensor(result);
  dil::concat::compute(x, dim, y);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(y.is_public_format() || check_tensor_own_whole_storage(result));
  dbl::comm::sync_shape_from_dil_to_aten(result, y);
  return result;
}

at::Tensor AtenIpexCPUDev::dil_cat(at::TensorList tensors, int64_t dim) {
  DEBUG("AtenIpexCPUDev::dil_cat\n");
  check_cat_no_zero_dim(tensors);
  dim = at::legacy_cat_wrap_dim(dim, tensors);
  std::vector<dil::tensor> x;
  at::Tensor tensors_contiguous[tensors.size()];
  for (auto i = 0; i < tensors.size(); i++) {
    IPEX_CHECK(!(tensors[i].dim() == 1 && tensors[i].sizes()[0] == 0),
      "Currently Mkldnn cat operators do not support empty tensor.");
    tensors_contiguous[i] = tensors[i].is_contiguous() ? tensors[i] : tensors[i].contiguous();

    dbl::comm::reorder_to_bf16_for_mix_prec(tensors_contiguous[i], true);

    x.push_back(dbl::comm::try_gen_dil_tensor(tensors_contiguous[i]));
  }
  dil::tensor y;
  dil::concat::compute(x, dim, y);
  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

std::vector<at::Tensor> AtenIpexCPUDev::dil_split_with_sizes(const at::Tensor& self, at::IntArrayRef split_sizes, int64_t dim) {
  DEBUG("AtenIpexCPUDev::dil_split_with_sizes\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(self, true);

  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  int64_t num_splits = split_sizes.size();
  std::vector<at::Tensor> splits(num_splits);
  std::vector<int32_t> sizes;
  for (auto i = 0; i < num_splits; i++) {
    auto length = split_sizes[i];
    IPEX_CHECK(length >= 0,
             "split_with_sizes expects split_sizes have only non-negative ",
             "entries, but got split_sizes=", split_sizes);
    sizes.push_back((int32_t)length);
  }

  dim = at::maybe_wrap_dim(dim, self.dim());
  auto y = dil::spliter::compute(x, sizes, dim, false);
  for (auto j = 0; j < num_splits; j++) {
    splits[j] = dbl::comm::gen_aten_tensor_by(std::move(y[j]));
  }
  return splits;
}


at::Tensor dil_as_strided(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset_) {

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      self.scalar_type() != at::kFloat || dbl::comm::try_gen_dil_tensor(self).is_public_format(),
      "Cannot set sizes and strides for DIL tensor with non-public format");

  // share storage
  auto* self_storage = self.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
  self_storage->data_ptr().unsafe_set_device(c10::Device(at::DeviceType::DPCPP));
  auto result = at::detail::make_tensor<IPEXTensorImpl>(self.storage(), at::DispatchKey::DPCPPTensorId);

  auto* _tensor_impl = (IPEXTensorImpl *)result.unsafeGetTensorImpl();
  _tensor_impl->copy_meta_info(self.unsafeGetTensorImpl());
  // When a tensor is chunked, the obtained chunked tensors do not share the version counter. 
  // We have copied the version counter in copy_meta_info and it is a workaround to reset the 
  // version counter here.
  // Note that when a tensor is sliced, PyTorch will call as_view which will copy the version 
  // counter to the sliced tensor. We do not need to handle it here.
  _tensor_impl->set_version_counter(0);
  _tensor_impl->copy_auto_grad(self.unsafeGetTensorImpl());

  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  _tensor_impl->set_strided(size, stride, storage_offset);
  return result;
}

at::Tensor& propagate_transposed_names(
    at::Tensor& result,
    const at::Tensor& other,
    int64_t dim0,
    int64_t dim1) {
  // Port from aten/src/ATen/native/TensorShape.cp
  if (other.has_names()) {
    auto names = other.names().vec();
    std::swap(names[dim0], names[dim1]);
    at::namedinference::propagate_names_if_nonempty(result, names);
  }
  return result;
}

at::Tensor AtenIpexCPUDev::dil_transpose(const at::Tensor & self, int64_t dim0, int64_t dim1) {
  DEBUG("AtenIpexCPUDev::dil_transpose\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(self, true);

  auto ndims = self.dim();
  dim0 = at::maybe_wrap_dim(dim0, ndims);
  dim1 = at::maybe_wrap_dim(dim1, ndims);
  if (dim0 == dim1) {
    return self;
  }

  // TODO: support transposing a blocked tensor
  // Even if DIL support transposing a blocked tensor, we have no place to
  // store a different desc for transposed view when storage are sharing.
  dbl::comm::reorder_to_public(self, /*remain_dtype=*/true);

  dil::tensor x = dbl::comm::try_gen_dil_tensor(self);
  IPEX_CHECK(x.ndims() > 0, "DNNL transpose cannot generate DNNL tensor for the input aten Tensor. input tensor dim: ", self.dim());

  auto trans_desc = x.get_desc().transpose(dim0, dim1);
  auto sizes = trans_desc.get_dims();
  auto strides = trans_desc.get_strides();
  auto result = dil_as_strided(self, sizes, strides, self.storage_offset());
  propagate_transposed_names(result, self, dim0, dim1);
  return result;
}

at::Tensor AtenIpexCPUDev::dil_slice(const at::Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step) {
  DEBUG("AtenIpexCPUDev::dil_slice\n");
  CHECK_DNNL_OP_PRE_COND(self);

  // TODO use weight TAG to decide whether to reorder or not
  dbl::comm::reorder_to_bf16_for_mix_prec(self, true);

  // Port from aten/src/ATen/native/TensorShape.cpp
  int64_t ndim = self.dim();
  if (ndim == 0) {
    AT_INDEX_ERROR("dil_slice() cannot be applied to a 0-dim tensor.");
  }
  dim = at::maybe_wrap_dim(dim, ndim);
  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  // TODO: support negative strides
  TORCH_CHECK(step > 0, "slice step must be positive");
  if (start < 0) {
    start += sizes[dim];
  }
  if (end < 0) {
    end += sizes[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= sizes[dim]) {
    start = sizes[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= sizes[dim]) {
    end = sizes[dim];
  }
  auto storage_offset = self.storage_offset() + start * strides[dim];
  auto len = end - start;
  sizes[dim] = (len + step - 1) / step;  // round-up
  strides[dim] *= step;

  auto result = dil_as_strided(self, sizes, strides, storage_offset);
  at::namedinference::propagate_names(result, self);
  return result;
}

std::vector<at::Tensor> AtenIpexCPUDev::dil_unbind(const at::Tensor &self, int64_t dim) {
  DEBUG("AtenIpexCPUDev::dil_unbind\n");

  dim = at::maybe_wrap_dim(dim, self.dim());
  int64_t size = dil_size(self, dim);
  std::vector<at::Tensor> tensors(size);
  for (int i = 0; i < size; i++) {
    tensors[i] = dil_select(self, dim, i);
  }
  return tensors;
}

std::vector<at::Tensor>AtenIpexCPUDev::dil_unbind(const at::Tensor& self, at::Dimname dim) {
  return dil_unbind(self, at::dimname_to_position(self, dim));
}

at::Tensor AtenIpexCPUDev::dil_select(const at::Tensor & self, int64_t dim, int64_t index) {
  DEBUG("AtenIpexCPUDev::dil_select\n");
  CHECK_DNNL_OP_PRE_COND(self);

  dbl::comm::reorder_to_bf16_for_mix_prec(self, true);

  // We do not support `select` a DIL tensor with blocked format
  dbl::comm::reorder_to_public(self, /*remain_dtype=*/true);

  // Port from aten/src/ATen/native/TensorShape.cpp
  int64_t ndim = self.dim();
  if (ndim == 0) {
    AT_INDEX_ERROR("select() cannot be applied to a 0-dim tensor.");
  }
  dim = at::maybe_wrap_dim(dim, ndim);
  auto size = dil_size(self, dim);
  if (index < -size || index >= size) {
    if (self.has_names() && self.names()[dim] != at::Dimname::wildcard()) {
      AT_INDEX_ERROR("select(): index ", index, " out of range for tensor of size ",
                     self.sizes(), " at dimension ", self.names()[dim]);
    }
    AT_INDEX_ERROR("select(): index ", index, " out of range for tensor of size ",
                   self.sizes(), " at dimension ", dim);
  }
  if (index < 0) {
    index += size;
  }
  if (self.is_sparse()) {
    IPEX_CHECK(false, "Got a sparse tensor in select. Should not reach here.");
    // return at::select(self, dim, index);
  }
  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  auto storage_offset = self.storage_offset() + index * strides[dim];
  sizes.erase(sizes.begin() + dim);
  strides.erase(strides.begin() + dim);
  auto result = dil_as_strided(self, sizes, strides, storage_offset);
  at::namedinference::propagate_names_except(result, self, {dim});
  return result;
}

at::Tensor alias_with_sizes_and_strides(
    const at::Tensor& self,
    const c10::IntArrayRef sizes,
    const c10::IntArrayRef strides) {

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dbl::comm::try_gen_dil_tensor(self).is_public_format(),
      "Cannot set sizes and strides for DIL tensor with non-public format");

  // share storage
  auto* self_storage = self.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();
  self_storage->data_ptr().unsafe_set_device(c10::Device(at::DeviceType::DPCPP));
  auto self_ = at::detail::make_tensor<IPEXTensorImpl>(self.storage(), at::DispatchKey::DPCPPTensorId);

  auto* _tensor_impl = (IPEXTensorImpl *)self_.unsafeGetTensorImpl();
  _tensor_impl->copy_meta_info(self.unsafeGetTensorImpl());
  _tensor_impl->copy_auto_grad(self.unsafeGetTensorImpl());

  auto storage_offset = self.storage_offset();
  _tensor_impl->set_strided(sizes, strides, storage_offset);

  at::namedinference::propagate_names(self_, self);
  return self_;
}

at::Tensor AtenIpexCPUDev::dil_view(const at::Tensor & self, at::IntArrayRef size) {
  DEBUG("AtenIpexCPUDev::dil_view\n");
  CHECK_DNNL_OP_PRE_COND(self);

  // We do not support reshaping (viewing) a DIL tensor with blocked format
  dbl::comm::reorder_to_public(self, /*remain_dtype=*/true);

  // Port from aten/src/ATen/native/TensorShape.cpp
  auto inferred_size = at::infer_size(size, self.numel());
  auto stride = at::detail::computeStride(self.sizes(),
                                          self.strides(),
                                          inferred_size);
  TORCH_CHECK(stride.has_value(), "view size is "
    "not compatible with input tensor's size and stride (at least one dimension"
    " spans across two contiguous subspaces). Use .reshape(...) instead.");
  auto stride_value = *stride;
  return alias_with_sizes_and_strides(self, inferred_size, stride_value);
}

at::Tensor AtenIpexCPUDev::dil__unsafe_view(const at::Tensor & self, at::IntArrayRef size) {
  DEBUG("AtenIpexCPUDev::dil__unsafe_view\n");
  return dil_view(self, size);
}

at::Tensor AtenIpexCPUDev::dil_select(const at::Tensor & self, at::Dimname dim, int64_t index) {
  return dil_select(self, at::dimname_to_position(self, dim), index);
}

at::Tensor _dil_narrow(const at::Tensor& self, int64_t dim, int64_t start, int64_t length) {
  // Port from aten/src/ATen/native/TensorShape.cpp
  TORCH_CHECK(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  auto cur_size = self.size(dim);
  if (start != cur_size) {  // start being the end is valid, but not a valid dim specification.
    start = at::maybe_wrap_dim(start, cur_size);
  }
  TORCH_CHECK(length >= 0 && start <= cur_size - length,
           "start (", start, ") + length (", length, ") exceeds dimension size (", cur_size, ").");
  return AtenIpexCPUDev::dil_slice(self, dim, start, start + length, 1);
}

std::vector<at::Tensor> AtenIpexCPUDev::dil_split(const at::Tensor& self, int64_t split_size, int64_t dim) {
  DEBUG("AtenIpexCPUDev::dil_split\n");
  // Port from aten/src/ATen/native/TensorShape.cpp
  TORCH_CHECK(self.dim() != 0, "split expects at least a 1-dimensional tensor");
  TORCH_CHECK(split_size >= 0,  "split expects split_size be non-negative, but got split_size=", split_size);
  
  CHECK_DNNL_OP_PRE_COND(self);
  dim = at::maybe_wrap_dim(dim, self.dim());
  int64_t dim_size = dil_size(self, dim);
  TORCH_CHECK(split_size > 0 || self.size(dim) == 0,
          "split_size can only be 0 if dimension size is 0, "
          "but got dimension size of ", dim_size);
  // if split_size is 0 and dimension size is 0, there is 1 split.
  int64_t num_splits = 1;
  if (split_size != 0) {
    // ensuring num_splits is at least 1 makes consistent the case where split_size > dim_size
    // (returns a single split).  We might want to error here, but keep it for BC.
    num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  }
  std::vector<at::Tensor> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - dim_size);

  for (int64_t i = 0; i < num_splits; ++i) {
    auto length = i < num_splits - 1 ? split_size : last_split_size;
    splits[i] = _dil_narrow(self, dim, i * split_size, length);
  }
  return splits;
}

at::Tensor AtenIpexCPUDev::dil_gelu(const at::Tensor& input) {
  DEBUG("AtenIpexCPUDev::dil_gelu\n");
  CHECK_DNNL_OP_PRE_COND(input);

  dbl::comm::reorder_to_bf16_for_mix_prec(input, true);

  dil::tensor x = dbl::comm::try_gen_dil_tensor(input);
  dil::tensor y;
  dil::eltwise_forward::compute(
      x, y, dil::algorithm::eltwise_gelu_tanh, dil::prop_kind::forward_training, /*alpha*/ 0.0);
  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

at::Tensor AtenIpexCPUDev::dil_gelu_backward(const at::Tensor& grad_output, const at::Tensor& input) {
  DEBUG("AtenIpexCPUDev::dil_gelu_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  CHECK_DNNL_OP_PRE_COND(input);

  dbl::comm::reorder_to_bf16_for_mix_prec(input, true);

  auto grad_output_contiguous = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  dbl::comm::reorder_to_bf16_for_mix_prec(grad_output_contiguous, true);

  dil::tensor x = dbl::comm::try_gen_dil_tensor(input);
  dil::tensor grady = dbl::comm::try_gen_dil_tensor(grad_output_contiguous);
  dil::tensor gradx;
  dil::eltwise_backward::compute(x, grady, gradx,
      dil::algorithm::eltwise_gelu_tanh, /*alpha*/ 0.0);
  return dbl::comm::gen_aten_tensor_by(std::move(gradx));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenIpexCPUDev::dil_native_layer_norm(
    const at::Tensor& X,
    const at::Tensor& gamma /* optional */,
    const at::Tensor& beta /* optional */,
    int64_t M,
    int64_t N,
    double eps) {
  DEBUG("AtenIpexCPUDev::dil_native_layer_norm\n");
  CHECK_DNNL_OP_PRE_COND(X);
  //It's a temporary solution to fall back to fp32 since bf16 layer_norm is not ready for dnnl path now.
  dbl::comm::reorder_to_dtype(X, at::kFloat);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(X);
  const dil::tensor scale = dbl::comm::try_gen_dil_tensor(gamma);
  const dil::tensor shift = dbl::comm::try_gen_dil_tensor(beta);
  int64_t i = 0;
  auto j = dil_size(X, 0);
  std::vector<int64_t> input_size;
  while(j <= M) {
    input_size.push_back(dil_size(X, i++));
    j *= dil_size(X, i);
  }
  input_size.push_back(N);
  auto src = x.reshape(input_size);
  dil::tensor y, mean, variance;
  dil::layer_normalization_forward::compute(
        src,
        scale,
        shift,
        y,
        mean,
        variance,
        eps);
  return std::make_tuple(
        dbl::comm::gen_aten_tensor_by(std::move(y)).reshape(X.sizes()),
        dbl::comm::gen_aten_tensor_by(std::move(mean)),
        dbl::comm::gen_aten_tensor_by(std::move(variance)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenIpexCPUDev::dil_native_layer_norm_backward(
    const at::Tensor& dY,
    const at::Tensor& X,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& gamma,
    int64_t M,
    int64_t N,
    std::array<bool, 3> grad_input_mask) {
  DEBUG("AtenIpexCPUDev::dil_native_layer_norm_backward\n");
  CHECK_DNNL_OP_PRE_COND(dY);
  CHECK_DNNL_OP_PRE_COND(X);
  //it's a temporary solution to fall back to fp32 since bf16 layer_norm is not ready for dnnl path now.
  dbl::comm::reorder_to_dtype(dY, at::kFloat);
  dbl::comm::reorder_to_dtype(X, at::kFloat);
  dil::tensor dy = dbl::comm::try_gen_dil_tensor(dY);
  dil::tensor x = dbl::comm::try_gen_dil_tensor(X);
  dil::tensor m = dbl::comm::try_gen_dil_tensor(mean);
  dil::tensor r = dbl::comm::try_gen_dil_tensor(rstd);
  dil::tensor g = dbl::comm::try_gen_dil_tensor(gamma);
  int64_t i = 0;
  auto j = dil_size(X, 0);
  std::vector<int64_t> input_size;
  while(j <= M) {
    input_size.push_back(dil_size(X, i++));
    j *= dil_size(X, i);
  }
  input_size.push_back(N);
  auto src = x.reshape(input_size);
  auto grady = dy.reshape(input_size);
  dil::tensor gradx, gradg, gradb;
  float eps = 1e-5;
  dil::layer_normalization_backward::compute(
        src, m, r, grady, g, gradx, gradg, gradb, eps);

  return std::make_tuple(
      dbl::comm::gen_aten_tensor_by(std::move(gradx)).reshape(X.sizes()),
      dbl::comm::gen_aten_tensor_by(std::move(gradg)),
      dbl::comm::gen_aten_tensor_by(std::move(gradb)));
}

at::Tensor AtenIpexCPUDev::dil_index_select(
    const at::Tensor & self,
    int64_t dim,
    const at::Tensor & index) {
  torch_ipex::reset_ipex_func_status();
  DEBUG("AtenIpexCPUDev::dil_index_select\n");
  IPEX_CHECK(
    self.device().type() == c10::DeviceType::DPCPP,
    "IPEX index select only work on DPCPP tensor");
  if (ShadeDataContext::isDilTensor(self) && ShadeDataContext::isTensorMixPrecision(self)) {
    dil::tensor& self_dil_storage = ShadeDataContext::getDilStorage(self);
    if (self_dil_storage.get_data_type() == dil::data_type::bf16) {
      return bf16::index_select(self, dim, index);
    }
    // TODO: We need add more LP here
  }

  torch_ipex::set_ipex_func_status(torch_ipex::IPEXFuncStatus::IPEX_FALLBACK);
  return at::Tensor();
}

at::Tensor AtenIpexCPUDev::dil_index(const at::Tensor & self, at::TensorList indices) {
  DEBUG("AtenIpexCPUDev::dil_index\n");
  torch_ipex::reset_ipex_func_status();

  IPEX_CHECK(
    self.device().type() == c10::DeviceType::DPCPP,
    "IPEX index only work on DPCPP tensor");
  if (ShadeDataContext::isDilTensor(self) && ShadeDataContext::isTensorMixPrecision(self)) {
    dil::tensor& self_dil_storage = ShadeDataContext::getDilStorage(self);
    if (self_dil_storage.get_data_type() == dil::data_type::bf16) {
      return bf16::index(self, indices);
    }
    // TODO: We need add more LP here
  }

  torch_ipex::set_ipex_func_status(torch_ipex::IPEXFuncStatus::IPEX_FALLBACK);
  return at::Tensor();
}

at::Tensor AtenIpexCPUDev::dil_shuffle(const at::Tensor & self, at::IntArrayRef view_shape, int64_t dim0, int64_t dim1) {
  DEBUG("AtenIpexCPUDev::dil_shuffle\n");
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexCPUDev::dil_shuffle", std::vector<c10::IValue>({self}), torch::autograd::Node::peek_at_next_sequence_nr());
#endif
  // NOTE: We do NOT add sanity checks here. Because PyTorch does not has shuffle operator. This dil operator is for fusion and the fusion logic
  // has more sanity checks. We found that there are some models use view + transpose + view to implement shuffle semantic. So IPEX will fuse these
  // operators a single shuffle.
  dil::tensor&& x = dbl::comm::try_gen_dil_tensor(self);
  dil::tensor y;
  auto group_dim = dim0 < dim1 ? dim0 : dim1;
  auto groups = view_shape[group_dim];
  dil::channel_shuffle_forward::compute(std::move(x), y, groups, group_dim);
  return dbl::comm::gen_aten_tensor_by(std::move(y));
}

std::tuple<at::Tensor,at::Tensor> AtenIpexCPUDev::dil__pack_padded_sequence(const at::Tensor & input, const at::Tensor & lengths, bool batch_first) {
  DEBUG("AtenIpexCPUDev::dil__pack_padded_sequence\n");
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexCPUDev::dil__pack_padded_sequence", std::vector<c10::IValue>({input, lengths}), torch::autograd::Node::peek_at_next_sequence_nr());
#endif
  torch_ipex::reset_ipex_func_status();

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.layout() == c10::kStrided);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(lengths.layout() == c10::kStrided);
  auto&& _ipex_input = bridge::shallowFallbackToCPUTensor(input);
  auto&& _ipex_lengths = bridge::shallowFallbackToCPUTensor(lengths);
  auto&& _ipex_result = at::_pack_padded_sequence(_ipex_input, _ipex_lengths, batch_first);

  // PyTorch requires the device of batch_size tensor is CPU, so IPEX does not covert the batch_size tensor(std::get<1>(_ipex_result)) to DPCPP
  return std::tuple<at::Tensor,at::Tensor>(
    bridge::shallowUpgradeToDPCPPTensor(std::get<0>(_ipex_result)),
    std::get<1>(_ipex_result));
}

at::Tensor& AtenIpexCPUDev::dil_copy_(
    at::Tensor & self,
    const at::Tensor & src,
    bool non_blocking) {
  DEBUG("AtenIpexCPUDev::dil_copy_\n");
  torch_ipex::reset_ipex_func_status();

  IPEX_CHECK(
    self.device().type() == c10::DeviceType::DPCPP &&
    src.device().type() == c10::DeviceType::DPCPP,
    "IPEX copy only work on DPCPP tensor");
  if (ShadeDataContext::isDilTensor(src) &&ShadeDataContext::isTensorMixPrecision(src)){
    IPEX_CHECK(check_tensor_own_whole_storage(self),  "IPEX copy only works while self tensor own the whole storage");
    auto dil_src = dbl::comm::try_gen_dil_tensor(src);
    IPEX_CHECK(dil_src.get_data_type() == dil::data_type::bf16)
    auto new_buffer_desc = dil_src.get_desc();
    dil::tensor dil_buffer{new_buffer_desc};
    dil_src.reorder_to(dil_buffer);
    dbl::comm::equip_dil_buffer(self, dil_buffer);
    return self;
  }
    // TODO: We need add more LP here
  torch_ipex::set_ipex_func_status(torch_ipex::IPEXFuncStatus::IPEX_FALLBACK);
  return self;
}

std::vector<at::Tensor> AtenIpexCPUDev::dil_rnn_layer(const at::Tensor& input, const at::Tensor& w1, const at::Tensor& w2,
    const at::Tensor& w3, const at::Tensor& w4, const at::Tensor& hx, const at::Tensor& cx, bool reverse, int64_t mode,
    int64_t hidden_size, int64_t num_layers, bool has_biases, bool train, bool bidirectional, at::IntArrayRef batch_sizes) {
  DEBUG("AtenIpexCPUDev::dil_rnn_layer\n");
  return dbl::rnn::mkldnn_rnn_layer(input, w1, w2, w3, w4, hx, cx, reverse, mode,
      hidden_size, num_layers, has_biases, train, bidirectional, batch_sizes);
}

std::vector<at::Tensor> AtenIpexCPUDev::dil_rnn_layer_backward(const at::Tensor& input, const at::Tensor& w1, const at::Tensor& w2,
    const at::Tensor& w3, const at::Tensor& w4, const at::Tensor& hx, const at::Tensor& cx, const at::Tensor& output, const at::Tensor& hy,
    const at::Tensor& cy, const at::Tensor& grad_output, const at::Tensor& grad_hy, const at::Tensor& grad_cy, bool reverse, int64_t mode,
    int64_t hidden_size, int64_t num_layers, bool has_biases, bool train, bool bidirectional, at::IntArrayRef batch_sizes) {
  DEBUG("AtenIpexCPUDev::dil_rnn_layer_backward\n");
  return dbl::rnn::mkldnn_rnn_layer_backward(input, w1, w2, w3, w4, hx, cx,
      output, hy, cy, grad_output, grad_hy, grad_cy, reverse, mode, hidden_size,
      num_layers, has_biases, train, bidirectional, batch_sizes);
}

at::Tensor AtenIpexCPUDev::dil_upsample_nearest1d(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales) {
  DEBUG("AtenIpexCPUDev::dil_upsample_nearest1d\n");
  CHECK_DNNL_OP_PRE_COND(self);
  return dbl::upsample::dil_upsample(self, output_size, dil::algorithm::resampling_nearest, scales);
}

at::Tensor AtenIpexCPUDev::dil_upsample_nearest1d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales) {
  DEBUG("AtenIpexCPUDev::dil_upsample_nearest1d_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  return dbl::upsample::dil_upsample_backward(grad_output, input_size, dil::algorithm::resampling_nearest, scales);
}

at::Tensor AtenIpexCPUDev::dil_upsample_nearest2d(const at::Tensor& input, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  DEBUG("AtenIpexCPUDev::dil_upsample_nearest2d\n");
  CHECK_DNNL_OP_PRE_COND(input);
  return dbl::upsample::dil_upsample(input, output_size, dil::algorithm::resampling_nearest, scales_h, scales_w);
}

at::Tensor AtenIpexCPUDev::dil_upsample_nearest2d_backward(const at::Tensor& grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  DEBUG("AtenIpexCPUDev::dil_upsample_nearest2d_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  return dbl::upsample::dil_upsample_backward(grad_output, input_size, dil::algorithm::resampling_nearest, scales_h, scales_w);
}

at::Tensor AtenIpexCPUDev::dil_upsample_nearest3d(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  DEBUG("AtenIpexCPUDev::dil_upsample_nearest3d\n");
  CHECK_DNNL_OP_PRE_COND(self);
  return dbl::upsample::dil_upsample(self, output_size, dil::algorithm::resampling_nearest, scales_d, scales_h, scales_w);
}

at::Tensor AtenIpexCPUDev::dil_upsample_nearest3d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  DEBUG("AtenIpexCPUDev::dil_upsample_nearest3d_backward\n");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  return dbl::upsample::dil_upsample_backward(grad_output, input_size, dil::algorithm::resampling_nearest, scales_d, scales_h, scales_w);
}

at::Tensor AtenIpexCPUDev::dil_upsample_linear1d(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales) {
  DEBUG("AtenIpexCPUDev::dil_upsample_linear1d\n");
  IPEX_CHECK(align_corners == false, "dil_upsample_linear1d not support align_corners mode yet");
  CHECK_DNNL_OP_PRE_COND(self);
  return dbl::upsample::dil_upsample(self, output_size, dil::algorithm::resampling_linear, scales);
}

at::Tensor AtenIpexCPUDev::dil_upsample_linear1d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales) {
  DEBUG("AtenIpexCPUDev::dil_upsample_linear1d_backward\n");
  IPEX_CHECK(align_corners == false, "dil_upsample_linear1d_backward not support align_corners mode yet");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  return dbl::upsample::dil_upsample_backward(grad_output, input_size, dil::algorithm::resampling_linear, scales);
}

at::Tensor AtenIpexCPUDev::dil_upsample_bilinear2d(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  DEBUG("AtenIpexCPUDev::dil_upsample_bilinear2d\n");
  IPEX_CHECK(align_corners == false, "dil_upsample_bilinear2d not support align_corners mode yet");
  CHECK_DNNL_OP_PRE_COND(self);
  return dbl::upsample::dil_upsample(self, output_size, dil::algorithm::resampling_linear, scales_h, scales_w);
}

at::Tensor AtenIpexCPUDev::dil_upsample_bilinear2d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  DEBUG("AtenIpexCPUDev::dil_upsample_bilinear2d_backward\n");
  IPEX_CHECK(align_corners == false, "dil_upsample_bilinear2d_backward not support align_corners mode yet");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  return dbl::upsample::dil_upsample_backward(grad_output, input_size, dil::algorithm::resampling_linear, scales_h, scales_w);
}

at::Tensor AtenIpexCPUDev::dil_upsample_trilinear3d(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  DEBUG("AtenIpexCPUDev::dil_upsample_trilinear3d\n");
  IPEX_CHECK(align_corners == false, "dil_upsample_trilinear3d not support align_corners mode yet");
  CHECK_DNNL_OP_PRE_COND(self);
  return dbl::upsample::dil_upsample(self, output_size, dil::algorithm::resampling_linear, scales_d, scales_h, scales_w);
}

at::Tensor AtenIpexCPUDev::dil_upsample_trilinear3d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  DEBUG("AtenIpexCPUDev::dil_upsample_trilinear3d_backward\n");
  IPEX_CHECK(align_corners == false, "dil_upsample_trilinear3d_backward not support align_corners mode yet");
  CHECK_DNNL_OP_PRE_COND(grad_output);
  return dbl::upsample::dil_upsample_backward(grad_output, input_size, dil::algorithm::resampling_linear, scales_d, scales_h, scales_w);
}

}  // namespace cpu
}  // namespace torch_ipex
