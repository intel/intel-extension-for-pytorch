
#include "Common.hpp"

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
                               std::string op_name, int64_t ops_id) {
  std::vector<std::vector<float>> inputs_min_max_values, outputs_min_max_values;
  for (auto i = 0; i < inputs.size(); i++) {
    inputs_min_max_values.push_back({inputs[i].abs().min().item<float>(), inputs[i].abs().max().item<float>()});
  } 
  for (auto j = 0; j < outputs.size(); j++) {
    outputs_min_max_values.push_back({outputs[j].abs().min().item<float>(), outputs[j].abs().max().item<float>()});
  }
  Int8OptConfig::get_config().insert_or_updata_observer(
      op_name, inputs_min_max_values, {}, outputs_min_max_values, ops_id);
}

void insert_or_updata_observer(const at::TensorList& inputs,
                               const at::TensorList& outputs,
                               const at::Tensor& weight,
                               std::string op_name, int64_t ops_id) {
  std::vector<std::vector<float>> inputs_min_max_values, outputs_min_max_values, weight_min_max_values={};
  for (auto i = 0; i < inputs.size(); i++) {
    inputs_min_max_values.push_back({inputs[i].abs().min().item<float>(), inputs[i].abs().max().item<float>()});
  } 
  // TODO: enable per_channel case
  /*
  for (auto k = 0; k < weight.size(0); k++) {
    weight_min_max_values.push_back({weight[k].abs().min().item<float>(), weight[k].abs().max().item<float>()});
  }*/
  
  weight_min_max_values.push_back({weight.abs().min().item<float>(), weight.abs().max().item<float>()});

  for (auto j = 0; j < outputs.size(); j++) {
    outputs_min_max_values.push_back({outputs[j].abs().min().item<float>(), outputs[j].abs().max().item<float>()});
  }
  Int8OptConfig::get_config().insert_or_updata_observer(
      op_name, inputs_min_max_values, weight_min_max_values, outputs_min_max_values, ops_id);
}

std::vector<std::vector<float>> get_int8_scales(std::vector<bool> i_uint8_used,
                                                     std::vector<bool> o_uint8_used,
                                                     const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_scales(i_uint8_used,
                                                          o_uint8_used, ops_id);
}

std::vector<float> get_int8_weight_scales(const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_weight_scales(ops_id);
}

bool get_int8_quantized_status(const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_quantized_status(ops_id);
}

std::tuple<bool, bool> get_int8_insert_quantized_status(const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_insert_quantized_status(ops_id);
}

} // namespce torch_ipex