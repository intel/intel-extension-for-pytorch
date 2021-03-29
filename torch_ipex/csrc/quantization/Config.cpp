#include "torch_ipex/csrc/utils.h"

#include <ATen/NativeFunctions.h>
#include <torch/csrc/autograd/function.h>

#include "Config.hpp"
// #include "Observer.hpp"

namespace torch_ipex {
using namespace int8;

void Int8OptConfig::insert_or_updata_observer(
    std::string op_name, std::vector<std::vector<float>> i_min_max_values,
    std::vector<std::vector<float>> w_min_max_values,
    std::vector<std::vector<float>> o_min_max_values, int64_t ops_id,
    std::vector<std::string> inputs_flow, std::vector<std::string> outputs_flow) {
  if (observers_.size() <= ops_id) {
    // this path is that to set int8 op's configure, using default configures if
    // user not set it. Note: weight's value only set onece.
    std::string observer_algorithm = "min_max";
    float averaging_constant = 0.01; // will be enabled for moving_averager_min_max
    std::string weight_granularity = "per_channel";
    const int nums_input = i_min_max_values.size();
    const int nums_output = o_min_max_values.size();
    std::vector<bool> inputs_dtype_uint8(nums_input, false);
    std::vector<bool> outputs_dtype_uint8(nums_output, false);
    bool quantized = true, pre_quantized = true, post_quantized = true;
    if (!indicators_.empty()) {
      observer_algorithm = indicators_[ops_id].get_indicator_algorithm();
      weight_granularity =
          indicators_[ops_id].get_indicator_weight_granularity();
      std::tie(inputs_dtype_uint8, outputs_dtype_uint8) =
          indicators_[ops_id].get_indicator_uint8_status();
      quantized = indicators_[ops_id].get_indicator_quantized_status();
      std::tie(pre_quantized, post_quantized) =
          indicators_[ops_id].get_indicator_insert_quantized_status();
    }
    Observer new_observer = {ops_id,
                             op_name,
                             i_min_max_values,
                             w_min_max_values,
                             o_min_max_values,
                             observer_algorithm,
                             averaging_constant,
                             weight_granularity,
                             inputs_dtype_uint8,
                             outputs_dtype_uint8,
                             quantized,
                             pre_quantized,
                             post_quantized,
                             inputs_flow,
                             outputs_flow};
    observers_.push_back(new_observer);
  } else {
    // user has set configure or have run one interation
    auto inputs_pre = observers_[ops_id].inputs_min_max_values;
    auto outputs_pre = observers_[ops_id].outputs_min_max_values;
    if (observers_[ops_id].algorithm == "min_max") {
      for (auto i = 0; i < i_min_max_values.size(); i++) {
        observers_[ops_id].inputs_min_max_values[i][0] =
            std::min(inputs_pre[i][0], i_min_max_values[i][0]);
        observers_[ops_id].inputs_min_max_values[i][1] =
            std::max(inputs_pre[i][1], i_min_max_values[i][1]);
      }
      for (auto j = 0; j < o_min_max_values.size(); j++) {
        observers_[ops_id].outputs_min_max_values[j][0] =
            std::min(outputs_pre[j][0], o_min_max_values[j][0]);
        observers_[ops_id].outputs_min_max_values[j][1] =
            std::max(outputs_pre[j][1], o_min_max_values[j][1]);
      }
    } else if (observers_[ops_id].algorithm == "moving_averager_min_max") {
      auto c = observers_[ops_id].averaging_constant;
      for (auto i = 0; i < i_min_max_values.size(); i++) {
        observers_[ops_id].inputs_min_max_values[i][0] =
            (1 - c) * inputs_pre[i][0] + c * i_min_max_values[i][0];
        observers_[ops_id].inputs_min_max_values[i][1] =
            (1 - c) * inputs_pre[i][1] + c * i_min_max_values[i][1];
      }
      for (auto j = 0; j < o_min_max_values.size(); j++) {
        observers_[ops_id].outputs_min_max_values[j][0] =
            (1 - c) * outputs_pre[j][0] + c * o_min_max_values[j][0];
        observers_[ops_id].outputs_min_max_values[j][1] =
            (1 - c) * outputs_pre[j][1] + c * o_min_max_values[j][1];
      }
    }
  }
}

void Int8OptConfig::clear_indicators() { indicators_.clear(); }

void Int8OptConfig::add_indicators() {
  indicators_.clear();
  // default used is s8
  for (auto i = 0; i < observers_.size(); i++) {
    std::vector<float> inputs_scale, outputs_scale;
    std::vector<float> weight_scales;
    std::vector<std::vector<float>> inputs_values =
        observers_[i].inputs_min_max_values;
    std::vector<std::vector<float>> outputs_values =
        observers_[i].outputs_min_max_values;
    std::vector<std::vector<float>> weight_values =
        observers_[i].weight_min_max_values;

    // for symmetric: s = 2max(|x_min|, x_max) / (Q_max - Q_min),
    // z = 0 for qint8 and z = 128 for quint8;
    // otherwise: s = (x_max - x_min) / (Q_max - Q_min),
    // z = Q_min - round(x_min / s).
    for (auto i = 0; i < inputs_values.size(); i++) {
      inputs_scale.push_back(
          std::max(std::abs(inputs_values[i][0]), inputs_values[i][1]) / 127.5);
    }
    for (auto j = 0; j < outputs_values.size(); j++) {
      outputs_scale.push_back(
          std::max(std::abs(outputs_values[j][0]), outputs_values[j][1]) / 127.5);
    }
    for (auto j = 0; j < weight_values.size(); j++) {
      weight_scales.push_back(
          std::max(std::abs(weight_values[j][0]), weight_values[j][1]) / 127.5);
    }
    // zero_points not used now, zero_points = 0 for s8 and 128 for u8.
    // zero_point = 128;
    Indicator new_indicator(
        observers_[i].id, observers_[i].name, observers_[i].algorithm,
        observers_[i].weight_granularity, inputs_scale, weight_scales, outputs_scale,
        observers_[i].inputs_dtype_uint8, observers_[i].outputs_dtype_uint8,
        observers_[i].quantized, observers_[i].pre_quantized, observers_[i].post_quantized,
        observers_[i].inputs_flow, observers_[i].outputs_flow);
    indicators_.push_back(new_indicator);
  }
  observers_.clear();
}

std::vector<std::vector<float>>
Int8OptConfig::get_indicator_scales(std::vector<bool> i_uint8_used,
                                    std::vector<bool> o_uint8_used,
                                    int64_t ops_id) {
  std::vector<float> inputs_scales, outputs_scales;
  std::tie(inputs_scales, outputs_scales) = indicators_[ops_id].get_indicator_scales();
  return  {inputs_scales, outputs_scales};
}

std::string Int8OptConfig::get_indicator_weight_granularity(const int64_t ops_id) {
  std::string weight_granularity = "per_channel";
  // user not set weight granularity, using default granularity
  if (indicators_.empty()) {
    return weight_granularity;
  }

  weight_granularity = indicators_[ops_id].get_indicator_weight_granularity();
  return weight_granularity;
}

float Int8OptConfig::get_indicator_weight_scale(const int64_t ops_id) {
  return indicators_[ops_id].get_indicator_weight_scales()[0];
}

at::Tensor& Int8OptConfig::get_indicator_weight_tensor_scale(const int64_t ops_id) {
  return weights_scales_[ops_id];
}

bool Int8OptConfig::get_indicator_quantized_status(const int64_t ops_id) {
  return indicators_[ops_id].get_indicator_quantized_status();
}

std::tuple<bool, bool> Int8OptConfig::get_indicator_insert_quantized_status(const int64_t ops_id) {
   return indicators_[ops_id].get_indicator_insert_quantized_status();
}

void Int8OptConfig::set_indicators(std::vector<Indicator> indicators) {
  // avoid to use copy assignment since the copy assignment for indicator with rw_mutex
  // have not been handdled properly
  indicators_.reserve(indicators.size());
  for (auto i: indicators){
    // if weight_granularity is per_channle, first cache the scales tensor for trace.
    if (i.get_indicator_weight_granularity() == "per_channel") {
      auto id = i.get_indicator_id();
      auto w_scales = i.get_indicator_weight_scales();
      auto casted_scale = at::tensor(w_scales, at::device(at::kCPU).dtype(at::kDouble));
      weights_scales_.emplace(id, casted_scale);
    }
    indicators_.emplace_back(i);
  }
}

std::vector<Indicator> Int8OptConfig::get_indicators() { return indicators_; }

int64_t Int8OptConfig::get_indicators_size() { return indicators_.size(); }

void Int8OptConfig::calibration_reset() { current_ops_id = 0; }

int64_t Int8OptConfig::fetch_and_add_ops_id() {
  int64_t ops_id = current_ops_id++;
  int64_t indicator_size = Int8OptConfig::get_config().get_indicators_size();
  if (current_ops_id == indicator_size)
    current_ops_id = 0;
  return ops_id;
}

thread_local int64_t Int8OptConfig::current_ops_id = 0;

} // namespace torch_ipex
