#include "AutoCast_utils.hpp"
#include <torch/torch.h>

#include <ATen/NativeFunctions.h>
#include <torch/csrc/autograd/function.h>

#include "Common.hpp"
#include "Config.hpp"
#include "torch_ipex/csrc/autocast_mode.h"
#include "torch_ipex/csrc/cpu/ExtendOPs.h"
namespace torch_ipex {
namespace autocast {
namespace int8 {

namespace {

using weakref_scales =
    c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>;
using val_scales = std::tuple<weakref_scales, at::Tensor>;
// TODO: zero_points cached
thread_local std::unordered_map<c10::TensorImpl*, val_scales> scales_casts;
// cach op tensors flow
// TODO: change the key work for slice or view.
using val_name = std::tuple<weakref_scales, std::string>;
thread_local std::unordered_map<c10::TensorImpl*, val_name>
    tensor_producer_name_map;

} // namespace

void clear_autocast_cache_int8() {
  scales_casts.clear();
  tensor_producer_name_map.clear();
}

params get_params(int64_t op_id) {
  struct params p;
  p.qparams = torch_ipex::get_int8_scales(op_id);
  std::tie(p.input_quantized_dtypes, p.output_quantized_dtypes) =
      torch_ipex::get_int8_quantized_dtypes(op_id);
  std::tie(p.inputs_quantized, p.outputs_quantized) =
      torch_ipex::get_int8_insert_quantized_status(op_id);
  return p;
}

void calibrate(
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& weights,
    const std::vector<at::Tensor>& outputs,
    const std::string& op_name,
    int64_t op_id,
    int op_type) {
  std::vector<std::string> op_inputs, op_outputs;
  for (auto i = 0; i < inputs.size(); i++) {
    auto it = tensor_producer_name_map.find(inputs[i].unsafeGetTensorImpl());
    if (it == tensor_producer_name_map.end()) {
      std::string op_input =
          op_name + std::to_string(op_id) + ".input" + std::to_string(i);
      op_inputs.push_back(op_input);
    } else {
      op_inputs.push_back(std::get<1>(it->second));
    }
  }

  if (op_type == OP_TYPE_INPLACE) {
    // Assuming only one input for inplace ops.
    TORCH_CHECK(
        inputs.size() == 1 && outputs.size() == 1,
        "The size of inputs and outputs should be one for inplace ops.");
    auto it = tensor_producer_name_map.find(inputs[0].unsafeGetTensorImpl());
    // Replacing tensor_producer_name_map using output info if the tensor has
    // been in tensor_producer_name_map, for example, conv2d -> relu, the
    // original tensor map is conv2d' output info, we need replace it using
    // relu's output info to following runinng data flow.
    std::string op_output =
        op_name + std::to_string(op_id) + ".output" + std::to_string(0);
    op_outputs.push_back(op_output);
    if (it != tensor_producer_name_map.end()) {
      it->second =
          val_name{weakref_scales(outputs[0].getIntrusivePtr()), op_output};
    } else {
      tensor_producer_name_map.emplace(
          outputs[0].unsafeGetTensorImpl(),
          val_name{weakref_scales(outputs[0].getIntrusivePtr()), op_output});
    }
    torch_ipex::insert_or_updata_observer(
        {}, {}, weights, op_name, op_id, op_inputs, op_outputs);
    return;
  }

  for (int j = 0; j < outputs.size(); j++) {
    std::string op_output =
        op_name + std::to_string(op_id) + ".output" + std::to_string(j);
    op_outputs.push_back(op_output);
    tensor_producer_name_map.emplace(
        outputs[j].unsafeGetTensorImpl(),
        val_name{weakref_scales(outputs[j].getIntrusivePtr()), op_output});
  }
  if (op_type == OP_TYPE_POOLING) {
    torch_ipex::insert_or_updata_observer(
        inputs, inputs, weights, op_name, op_id, op_inputs, op_outputs);
  } else {
    torch_ipex::insert_or_updata_observer(
        inputs, outputs, weights, op_name, op_id, op_inputs, op_outputs);
  }
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> insert_q_dq_inputs(
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& weights,
    const std::vector<quant_utils::TensorQuantizationParams>& inputs_qparams,
    const std::vector<at::ScalarType>& inputs_quantized_dtypes,
    const std::vector<bool>& inputs_quantized,
    int64_t op_id) {
  std::vector<at::Tensor> r_inputs, r_weights;
  // if a op's one input is inserted a quantizer, the weights' quantizer will be
  // inserted.
  bool weights_quantized = false;
  for (int i = 0; i < inputs.size(); i++) {
    if (inputs_quantized[i]) {
      bool bf16 = inputs[i].scalar_type() == at::kBFloat16;
      auto input_q = at::quantize_per_tensor(
          bf16 ? inputs[i].to(at::kFloat) : inputs[i],
          inputs_qparams[i].scale,
          inputs_qparams[i].zero_point,
          inputs_quantized_dtypes[i]);
      r_inputs.push_back(
          bf16 ? input_q.dequantize().to(at::kBFloat16) : input_q.dequantize());
      weights_quantized = true;
    } else {
      r_inputs.push_back(inputs[i]);
    }
  }
  // if has weight and need insert quantizer before weight.
  auto weight_granularity = torch_ipex::get_int8_weight_granularity(op_id);
  for (int i = 0; i < weights.size(); i++) {
    if (weights_quantized) {
      bool bf16 = weights[i].scalar_type() == at::kBFloat16;
      if (weight_granularity == "per_channel") {
        auto& weight_scale = torch_ipex::get_int8_weight_tensor_scale(op_id);
        auto zero_points =
            at::zeros(weight_scale[i].numel(), at::dtype(at::kLong));
        auto weight_q = at::quantize_per_channel(
            bf16 ? weights[i].to(at::kFloat) : weights[i],
            weight_scale[i],
            zero_points,
            0,
            at::kQInt8);
        r_weights.push_back(
            bf16 ? weight_q.dequantize().to(at::kBFloat16)
                 : weight_q.dequantize());
      } else {
        std::vector<float> w_scale = torch_ipex::get_int8_weight_scale(op_id);
        auto weight_q = at::quantize_per_tensor(
            bf16 ? weights[i].to(at::kFloat) : weights[i],
            w_scale[i],
            0,
            at::kQInt8);
        r_weights.push_back(
            bf16 ? weight_q.dequantize().to(at::kBFloat16)
                 : weight_q.dequantize());
      }
    } else {
      r_weights.push_back(weights[i]);
    }
  }
  return std::make_tuple(r_inputs, r_weights);
}

std::vector<at::Tensor> insert_q_dq_outputs(
    const std::vector<at::Tensor>& outputs,
    const std::vector<quant_utils::TensorQuantizationParams>& outputs_qparams,
    const std::vector<at::ScalarType>& outputs_quantized_dtypes,
    const std::vector<bool>& outputs_quantized) {
  std::vector<at::Tensor> r_outputs;
  for (int i = 0; i < outputs.size(); i++) {
    if (outputs_quantized[i]) {
      bool bf16 = outputs[i].scalar_type() == at::kBFloat16;
      auto output_q = at::quantize_per_tensor(
          bf16 ? outputs[i].to(at::kFloat) : outputs[i],
          outputs_qparams[i].scale,
          outputs_qparams[i].zero_point,
          outputs_quantized_dtypes[i]);
      r_outputs.push_back(
          bf16 ? output_q.dequantize().to(at::kBFloat16)
               : output_q.dequantize());
    } else {
      r_outputs.push_back(outputs[i]);
    }
  }
  return r_outputs;
}
} // namespace int8
} // namespace autocast
} // namespace torch_ipex
