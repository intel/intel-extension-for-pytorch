#include "torch_ipex/csrc/cpu/FusionOPs.h"

#include <ATen/Context.h>
#include <ATen/CPUGenerator.h>
#include <ATen/InferSize.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/record_function.h>

#include <limits>

#include "torch_ipex/csrc/cpu/int8/Config.h"
#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/ipex_tensor_impl.h"
#include "torch_ipex/csrc/utils.h"
#include "torch_ipex/csrc/cpu/DevOPs.h"
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
    const dil::attr_t& op_attr,
    const std::string& op_name = "Convolution_Relu") {
  dil::tensor dil_input;
  dil::tensor dil_weight;
  c10::optional<dil::tensor> dil_bias{c10::nullopt};
  // for int8 path, input always acbd format which is non-contiguous, .contiguous() will reorder to fp32
  auto src_dil_type = dbl::comm::try_gen_dil_tensor(input).get_data_type();
  auto input_contiguous = (src_dil_type == dil::data_type::u8 || src_dil_type == dil::data_type::s8
                           || input.is_contiguous()) ? input : input.contiguous();
  auto weight_dil_type = dbl::comm::try_gen_dil_tensor(weight).get_data_type();
  auto weight_contiguous = (weight_dil_type == dil::data_type::s8 || weight.is_contiguous()) ? weight : weight.contiguous();

  bool quantized = false;
  std::vector<float> output_scale = {};
  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    int64_t num_ops_id = Int8OptConfig::fetch_and_add_ops_id();
    quantized = dbl::comm::get_int8_quantized_status(num_ops_id);
    std::vector<std::vector<float>> scales = dbl::comm::get_int8_scales(
        {input}, /*  uint8_used for output*/ false, num_ops_id);
    if (quantized) {
      output_scale.push_back(scales[1][0]);
      dbl::comm::reorder_to_int8_for_mix_prec(input_contiguous, scales[0]);
      dbl::comm::reorder_to_int8_for_mix_prec(weight_contiguous, {});
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
      dbl::comm::reorder_to_dtype(weight, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input_contiguous);
    dbl::comm::reorder_to_bf16_for_mix_prec(weight_contiguous);
  }

  dil_input = try_gen_dil_tensor(input_contiguous);
  dbl::conv::prepack_conv_weights(
    input_contiguous,
    dil_input,
    weight_contiguous,
    stride,
    padding,
    dilation,
    groups);
  dil_weight = try_gen_dil_tensor(weight_contiguous);

  if (bias.defined()) {
    auto bias_contiguous = bias.is_contiguous() ? bias : bias.contiguous();
    if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
      if (quantized) {
        auto src = dbl::comm::try_gen_dil_storage(bias_contiguous);
        auto src_type = src.get_data_type();
        if (src_type != dil::data_type::s32) {
          auto dst_desc = src.get_desc().to_type(dil::data_type::s32);
          auto bias_scales = dil_weight.get_scale();
          for (auto &scale : bias_scales) { scale *= dil_input.get_scale()[0];  }
          dbl::comm::reorder_to_desc(bias_contiguous, dst_desc, bias_scales);
        }
      } else {
        dbl::comm::reorder_to_dtype(bias_contiguous, at::kFloat);
      }
    } else {
      dbl::comm::reorder_to_bf16_for_mix_prec(bias_contiguous);
    }
    dil_bias = dbl::comm::try_gen_dil_tensor(bias_contiguous);
  }

  dil::tensor dil_output = dbl::conv::convolution_impl(
    dil_input,
    dil_weight,
    dil_bias,
    padding,
    stride,
    dilation,
    groups,
    op_attr,
    output_scale);

  auto aten_output = dbl::comm::gen_aten_tensor_by(std::move(dil_output));
  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer({input_contiguous}, {aten_output}, op_name,
                              Int8OptConfig::fetch_and_add_ops_id());
  }
  return aten_output;
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
    const dil::attr_t& attr,
    const std::string& op_name) {
  dil::tensor dil_input;
  dil::tensor dil_weight;
  dil::tensor dil_output;
  c10::optional<dil::tensor> dil_bias{c10::nullopt};

  // for int8 path, input always acbd format which is non-contiguous, .contiguous() will reorder to fp32
  auto src_dil_type = dbl::comm::try_gen_dil_tensor(input).get_data_type();
  auto input_contiguous = (src_dil_type == dil::data_type::u8 || src_dil_type == dil::data_type::s8
                           || input.is_contiguous()) ? input : input.contiguous();
  auto weight_dil_type = dbl::comm::try_gen_dil_tensor(weight).get_data_type();
  auto weight_contiguous = (weight_dil_type == dil::data_type::s8 || weight.is_contiguous()) ? weight : weight.contiguous();
  auto ouput_dil_type = dbl::comm::try_gen_dil_tensor(accumu).get_data_type();
  auto output_contiguous = (ouput_dil_type == dil::data_type::u8 || ouput_dil_type == dil::data_type::s8 || accumu.is_contiguous()) ? accumu : accumu.contiguous();

  bool quantized = false;
  std::vector<float> output_scale = {};
  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    int64_t num_ops_id = Int8OptConfig::fetch_and_add_ops_id();
    quantized = dbl::comm::get_int8_quantized_status(num_ops_id);
    std::vector<std::vector<float>> scales = dbl::comm::get_int8_scales(
        {input}, /*  uint8_used for output*/ false, num_ops_id);
    if (quantized) {
      output_scale.push_back(scales[1][0]);
      dbl::comm::reorder_to_int8_for_mix_prec(input_contiguous, scales[0]);
      dbl::comm::reorder_to_int8_for_mix_prec(weight_contiguous, {});
    } else {
      dbl::comm::reorder_to_dtype(input_contiguous, at::kFloat);
      dbl::comm::reorder_to_dtype(weight_contiguous, at::kFloat);
      // ouput may a int8 tensor, should reorder to fp32
      dbl::comm::reorder_to_dtype(output_contiguous, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input_contiguous);
    dbl::comm::reorder_to_bf16_for_mix_prec(weight_contiguous);
    dbl::comm::reorder_to_bf16_for_mix_prec(output_contiguous);
  }

  dil_input = try_gen_dil_tensor(input_contiguous);
  dil_output = try_gen_dil_tensor(output_contiguous);

  dbl::conv::prepack_conv_weights(
    input_contiguous,
    dil_input,
    weight_contiguous,
    stride,
    padding,
    dilation,
    groups);
  dil_weight = try_gen_dil_tensor(weight_contiguous);

  if (bias.defined()) {
    auto bias_contiguous = bias.is_contiguous() ? bias : bias.contiguous();
    if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
      if (quantized) {
        auto src = dbl::comm::try_gen_dil_storage(bias_contiguous);
        auto src_type = src.get_data_type();
        if (src_type != dil::data_type::s32) {
          auto dst_desc = src.get_desc().to_type(dil::data_type::s32);
          auto bias_scales = dil_weight.get_scale();
          for (auto &scale : bias_scales) { scale *= dil_input.get_scale()[0];  }
          dbl::comm::reorder_to_desc(bias_contiguous, dst_desc, bias_scales);
        }
      } else {
        dbl::comm::reorder_to_dtype(bias_contiguous, at::kFloat);
      }
    } else {
      dbl::comm::reorder_to_bf16_for_mix_prec(bias_contiguous);
    }
    dil_bias = dbl::comm::try_gen_dil_tensor(bias_contiguous);
  }

  dbl::conv::convolution_inplace_impl(
    dil_input,
    dil_weight,
    dil_bias,
    dil_output,
    padding,
    stride,
    dilation,
    groups,
    attr,
    output_scale);

  dbl::comm::equip_dil_buffer(accumu, dil_output);
  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer({input_contiguous}, {accumu}, op_name,
                              Int8OptConfig::fetch_and_add_ops_id());
  }
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
    dil::attr_t::fuse_relu(),
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
    dil::attr_t::fuse_sum(scale),
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
  at::Tensor& output = dil_convolution_inplace_fusion(
    input,
    weight,
    bias,
    accumu,
    stride,
    padding,
    dilation,
    groups,
    dil::attr_t::residual(scale),
    "Convolution_Sum_Relu");
  // if the next operator is convolution, u8 output can get a better performance than s8, so always convert
  // accumu's dil tensor to u8 data type.
  ShadeDataContext *shade_data_contex = (ShadeDataContext*)(output.storage().data_ptr().get_context());
  if (shade_data_contex->mix_prec_type == MIX_PREC_TYPE::MIX_INT8_FP32) {
    shade_data_contex->dil_tensor->to_type(dil::data_type::u8);
  }
  return output;
}

at::Tensor AtenIpexJITDev::dil_linear_fuse_eltwise(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const dil::attr_t& attr) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_linear_fuse_eltwise", std::vector<c10::IValue>({self, weight, bias}), torch::autograd::Node::peek_at_next_sequence_nr());
#endif
  return AtenIpexCPUDev::dil_linear(self, weight, bias, attr);
}

}  // namespace cpu
}  // namespace torch_ipex
