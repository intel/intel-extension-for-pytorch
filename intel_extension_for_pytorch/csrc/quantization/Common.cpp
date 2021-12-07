
#include "Common.hpp"

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <c10/util/Exception.h>

#include "Config.hpp"
#include "auto_opt_config.hpp"

namespace torch_ipex {

bool check_int8_calibration() {
  return AutoOptConfig::singleton().get_int8_calibration();
}

void insert_or_updata_observer(
    const at::TensorList& inputs,
    const at::TensorList& outputs,
    const at::TensorList& weights,
    std::string op_name,
    int64_t ops_id,
    std::vector<std::string> op_inputs,
    std::vector<std::string> op_outputs) {
  std::vector<std::vector<float>> inputs_min_max_values, outputs_min_max_values;
  std::vector<std::vector<std::vector<float>>> weights_min_max_values;
  for (auto i = 0; i < inputs.size(); i++) {
    inputs_min_max_values.push_back(
        {inputs[i].min().item<float>(), inputs[i].max().item<float>()});
  }

  for (auto j = 0; j < outputs.size(); j++) {
    outputs_min_max_values.push_back(
        {outputs[j].min().item<float>(), outputs[j].max().item<float>()});
  }
  if (weights.size() > 0) {
    auto weight_granularity =
        Int8OptConfig::get_config().get_indicator_weight_granularity(ops_id);
    if (Int8OptConfig::get_config().get_indicators_size() == 0 &&
        op_name == "embedding_bag") {
      weight_granularity = "per_tensor";
    }
    if (weight_granularity == "per_channel") {
      for (auto k = 0; k < weights.size(); k++) {
        auto w = weights[k];
        std::vector<std::vector<float>> min_max_values;
        for (int l = 0; l < w.size(0); l++) {
          min_max_values.push_back(
              {w[l].min().item<float>(), w[l].max().item<float>()});
        }
        weights_min_max_values.push_back(min_max_values);
      }
    } else {
      for (auto k = 0; k < weights.size(); k++) {
        std::vector<std::vector<float>> min_max_values;
        min_max_values.push_back(
            {weights[k].min().item<float>(), weights[k].max().item<float>()});
        weights_min_max_values.push_back(min_max_values);
      }
    }
  }
  Int8OptConfig::get_config().insert_or_updata_observer(
      op_name,
      inputs_min_max_values,
      weights_min_max_values,
      outputs_min_max_values,
      ops_id,
      op_inputs,
      op_outputs);
}

std::vector<std::vector<quant_utils::TensorQuantizationParams>> get_int8_scales(
    const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_scales(ops_id);
}

std::string get_int8_weight_granularity(const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_weight_granularity(ops_id);
}

std::vector<float> get_int8_weight_scale(const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_weight_scale(ops_id);
}

std::vector<at::Tensor>& get_int8_weight_tensor_scale(const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_weight_tensor_scale(ops_id);
}

std::tuple<std::vector<bool>, std::vector<bool>>
get_int8_insert_quantized_status(const int64_t ops_id) {
  return Int8OptConfig::get_config().get_indicator_insert_quantized_status(
      ops_id);
}

std::tuple<std::vector<at::ScalarType>, std::vector<at::ScalarType>>
get_int8_quantized_dtypes(const int64_t ops_id) {
  std::vector<std::string> input_dtypes, output_dtypes;
  std::tie(input_dtypes, output_dtypes) =
      Int8OptConfig::get_config().get_indicator_quantized_dtypes(ops_id);
  std::vector<at::ScalarType> x_dtypes, y_dtypes;
  for (auto& type : input_dtypes) {
    auto x_dtype = (type == "int8") ? at::kQInt8 : at::kQUInt8;
    x_dtypes.push_back(x_dtype);
  }
  for (auto& type : output_dtypes) {
    auto y_dtype = (type == "int8") ? at::kQInt8 : at::kQUInt8;
    y_dtypes.push_back(y_dtype);
  }
  return std::make_tuple(x_dtypes, y_dtypes);
}

} // namespace torch_ipex
