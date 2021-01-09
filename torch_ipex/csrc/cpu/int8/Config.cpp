#include "utils.h"
#include "Config.h"
#include "cpu/int8/quantization/Observer.h"
#include <mutex>
#include <thread>

namespace torch_ipex {
using namespace torch_ipex::cpu::lp::int8;

void Int8OptConfig::insert_or_updata_observer(
    std::string op_name, std::vector<std::vector<float>> i_min_max_values,
    std::vector<std::vector<float>> o_min_max_values, int64_t ops_id) {
  if (observers_.size() <= ops_id) {
      // this path is that to set int8 op's configure, using default configures if user not set it.
      std::string observer_algorithm = "min_max";
      float averaging_constant = 0.01; // will be enabled for moving_averager_min_max
      std::string weight_granularity = "per_channel";
      const int nums_input = i_min_max_values.size();
      const int nums_output = o_min_max_values.size();
      std::vector<bool> inputs_dtype_uint8(nums_input, false);
      std::vector<bool> outputs_dtype_uint8(nums_output, false);
      bool quantized = true;
      if (!indicators_.empty()) {
        observer_algorithm = indicators_[ops_id].get_indicator_algorithm();
        weight_granularity = indicators_[ops_id].get_indicator_weight_granularity();
        std::tie(inputs_dtype_uint8, outputs_dtype_uint8) = indicators_[ops_id].get_indicator_uint8_status();
        quantized = indicators_[ops_id].get_indicator_quantized_status();
      }
      Observer new_observer = {ops_id, op_name, i_min_max_values, o_min_max_values, observer_algorithm,
          averaging_constant, weight_granularity, inputs_dtype_uint8, outputs_dtype_uint8, quantized};
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
    std::vector<std::vector<float>> inputs_values =
        observers_[i].inputs_min_max_values;
    std::vector<std::vector<float>> outputs_values =
        observers_[i].outputs_min_max_values;

    for (auto i = 0; i < inputs_values.size(); i++) {
      inputs_scale.push_back(
          127.5 / std::max(std::abs(inputs_values[i][0]), inputs_values[i][1]));
    }
    for (auto j = 0; j < outputs_values.size(); j++) {
      outputs_scale.push_back(127.5 / std::max(std::abs(outputs_values[j][0]),
                                               outputs_values[j][1]));
    }
    // zero_points not used now, zero_points = 0 for u8 and 128 for s8.
    // zero_point = 128;
    Indicator new_indicator(
        observers_[i].id, observers_[i].name, observers_[i].algorithm,
        observers_[i].weight_granularity, inputs_scale, outputs_scale,
        observers_[i].inputs_dtype_uint8, observers_[i].outputs_dtype_uint8,
        observers_[i].quantized);
    indicators_.push_back(new_indicator);
  }
  observers_.clear();
}

std::vector<std::vector<float>>
Int8OptConfig::get_indicator_scales(std::vector<bool> i_uint8_used,
                                    std::vector<bool> o_uint8_used,
                                    int64_t ops_id) {
  std::vector<float> inputs_scale, outputs_scale;
  std::vector<bool> inputs_uint8_used, outputs_uint8_used;
  std::tie(inputs_uint8_used, outputs_uint8_used) =
      indicators_[ops_id].get_indicator_uint8_status();
  std::tie(inputs_scale, outputs_scale) =
      indicators_[ops_id].get_indicator_scales();
  bool scale_update = false;
  for (auto i = 0; i < i_uint8_used.size(); i++) {
    if (!inputs_uint8_used[i] && i_uint8_used[i]) {
      // update zero_point and scales
      inputs_scale[i] /= 127.5;
      inputs_scale[i] *= 255.5;
      scale_update = true;
    } else if (inputs_uint8_used[i] && !i_uint8_used[i]) {
      // update zero_point and scales
      inputs_scale[i] /= 255.5;
      inputs_scale[i] *= 127.5;
      scale_update = true;
    }
  }
  for (auto j = 0; j < o_uint8_used.size(); j++) {
    if (!outputs_uint8_used[j] && o_uint8_used[j]) {
      // update zero_point and scales
      outputs_scale[j] /= 127.5;
      outputs_scale[j] *= 255.5;
      scale_update = true;
    } else if (outputs_uint8_used[j] && !o_uint8_used[j]) {
      // update zero_point and scales
      outputs_scale[j] /= 255.5;
      outputs_scale[j] *= 127.5;
      scale_update = true;
    }
  }
  if (scale_update) {
    indicators_[ops_id].set_indicator_scales(inputs_scale, outputs_scale);
    indicators_[ops_id].set_indicator_uint8_status(inputs_uint8_used,
                                                   outputs_uint8_used);
  }
  std::vector<std::vector<float>> input_output_scale = {inputs_scale,
                                                        outputs_scale};
  return input_output_scale;
}

bool Int8OptConfig::get_indicator_quantized_status(int64_t ops_id) {
  return indicators_[ops_id].get_indicator_quantized_status();
}
void Int8OptConfig::set_indicators(std::vector<Indicator> indicators) {
  indicators_ = indicators;
}

std::vector<Indicator> Int8OptConfig::get_indicators() { return indicators_; }

void Int8OptConfig::calibration_reset() { current_ops_id = 0; }

int64_t Int8OptConfig::fetch_and_add_ops_id() {
  if (current_ops_id == indicators_.size())
    current_ops_id = 0;
  int64_t ops_id = current_ops_id++;
  return ops_id;
}

thread_local int64_t Int8OptConfig::current_ops_id = 0;
} // namespace torch_ipex