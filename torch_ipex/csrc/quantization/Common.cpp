
#include "Common.hpp"

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <c10/util/Exception.h>

#include "torch_ipex/csrc/auto_opt_config.hpp"
#include "Config.hpp"


namespace torch_ipex {

bool check_int8_calibration() {
  return AutoOptConfig::singleton().get_int8_calibration();
}

void insert_or_updata_observer(const at::TensorList &inputs,
                               const at::TensorList &outputs,
                               std::string op_name,
                               int64_t ops_id,
                               std::vector<std::string> inputs_flow,
                               std::vector<std::string> outputs_flow) {
  std::vector<std::vector<float>> inputs_min_max_values, outputs_min_max_values;
  for (auto i = 0; i < inputs.size(); i++) {
    inputs_min_max_values.push_back({inputs[i].min().item<float>(), inputs[i].max().item<float>()});
  } 
  for (auto j = 0; j < outputs.size(); j++) {
    outputs_min_max_values.push_back({outputs[j].min().item<float>(), outputs[j].max().item<float>()});
  }
  Int8OptConfig::get_config().insert_or_updata_observer(
      op_name, inputs_min_max_values, {}, outputs_min_max_values, ops_id, inputs_flow, outputs_flow);
}

void insert_or_updata_observer(const at::TensorList& inputs,
                               const at::TensorList& outputs,
                               const at::Tensor& weight,
                               std::string op_name, int64_t ops_id,
                               std::vector<std::string> op_inputs,
                               std::vector<std::string> op_outputs ) {
  std::vector<std::vector<float>> inputs_min_max_values, outputs_min_max_values, weight_min_max_values={};
  for (auto i = 0; i < inputs.size(); i++) {
    inputs_min_max_values.push_back({inputs[i].min().item<float>(), inputs[i].max().item<float>()});
  } 

  if (Int8OptConfig::get_config().get_indicator_weight_granularity(ops_id) == "per_channel") {
    for (auto k = 0; k < weight.size(0); k++) {
      weight_min_max_values.push_back({weight[k].min().item<float>(), weight[k].max().item<float>()});
    }
  } else {
    weight_min_max_values.push_back({weight.min().item<float>(), weight.max().item<float>()});
  }

  for (auto j = 0; j < outputs.size(); j++) {
    outputs_min_max_values.push_back({outputs[j].min().item<float>(), outputs[j].max().item<float>()});
  }
  Int8OptConfig::get_config().insert_or_updata_observer(
      op_name, inputs_min_max_values, weight_min_max_values, outputs_min_max_values, ops_id,
      op_inputs, op_outputs);
}

std::vector<std::vector<float>> get_int8_scales(std::vector<bool> i_uint8_used,
                                                std::vector<bool> o_uint8_used,
                                                const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_scales(i_uint8_used,
                                                          o_uint8_used, ops_id);
}

std::string get_int8_weight_granularity(const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_weight_granularity(ops_id);
}

float get_int8_weight_scale(const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_weight_scale(ops_id);
}

at::Tensor& get_int8_weight_tensor_scale(const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_weight_tensor_scale(ops_id);
}

std::tuple<std::vector<bool>, std::vector<bool>> get_int8_insert_quantized_status(const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_insert_quantized_status(ops_id);
}

} // namespce torch_ipex
