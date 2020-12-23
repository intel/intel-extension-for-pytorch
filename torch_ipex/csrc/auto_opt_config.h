#pragma once

#include "quantization/Observer.h"
#include "utils.h"

namespace torch_ipex {

using namespace torch_ipex::cpu::lp::int8;

class AutoOptConfig {
public:
  static AutoOptConfig& singleton() {
    static AutoOptConfig auto_opt_conf;
    return auto_opt_conf;
  }

public:
  inline void set_auto_dnnl(bool auto_dnnl) {
    auto_dnnl_ = auto_dnnl;
  }
  inline bool get_auto_dnnl() {
    return auto_dnnl_;
  }

  inline void set_jit_fuse(bool jit_fuse) {
    jit_fuse_ = jit_fuse;
  }

  inline bool get_jit_fuse() {
    return jit_fuse_;
  }

  // bf16
  inline void set_mix_bf16_fp32(bool value) {
    mix_bf16_fp32_ = value;
  }
  inline bool get_mix_bf16_fp32() {
    return mix_bf16_fp32_;
  }

  inline void set_train(bool value) {
    train_ = value;
  }
  inline bool get_train() {
    return train_;
  }

  // int8
  inline void set_mix_int8_fp32(bool value) {
    mix_int8_fp32_ = value;
  }
  inline bool get_mix_int8_fp32() {
    return mix_int8_fp32_;
  }

  inline void set_int8_calibration(bool value) {
    calibration_step_ = value;
  }
  inline bool get_int8_calibration() {
    return calibration_step_;
  }

  inline void insert_or_updata_observer(std::string op_name,
    std::vector<std::vector<float>> i_min_max_values, std::vector<std::vector<float>> o_min_max_values) {
    num_ops_id++;
    if (observers_.size() < num_ops_id) {
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
        observer_algorithm = indicators_[num_ops_id - 1].get_indicator_algorithm();
        weight_granularity = indicators_[num_ops_id - 1].get_indicator_weight_granularity();
        std::tie(inputs_dtype_uint8, outputs_dtype_uint8) = indicators_[num_ops_id - 1].get_indicator_uint8_status();
        quantized = indicators_[num_ops_id - 1].get_indicator_quantized_status();
      }
      Observer new_observer = {num_ops_id - 1, op_name, i_min_max_values, o_min_max_values, observer_algorithm,
          averaging_constant, weight_granularity, inputs_dtype_uint8, outputs_dtype_uint8, quantized};
      observers_.push_back(new_observer);
    } else {
      // user has set configure or have run one interation
      auto inputs_pre = observers_[num_ops_id - 1].inputs_min_max_values;
      auto outputs_pre = observers_[num_ops_id - 1].outputs_min_max_values;
      if (observers_[num_ops_id - 1].algorithm == "min_max") {
        for (auto i = 0; i < i_min_max_values.size(); i++) {
          observers_[num_ops_id - 1].inputs_min_max_values[i][0] = std::min(inputs_pre[i][0], i_min_max_values[i][0]);
          observers_[num_ops_id - 1].inputs_min_max_values[i][1] = std::max(inputs_pre[i][1], i_min_max_values[i][1]);
        }
        for (auto j = 0; j < o_min_max_values.size(); j++) {
          observers_[num_ops_id - 1].outputs_min_max_values[j][0]= std::min(outputs_pre[j][0], o_min_max_values[j][0]);
          observers_[num_ops_id - 1].outputs_min_max_values[j][1] = std::max(outputs_pre[j][1], o_min_max_values[j][1]);
        }
      } else if(observers_[num_ops_id -1].algorithm == "moving_averager_min_max"){
        auto c = observers_[num_ops_id - 1].averaging_constant;
        for (auto i = 0; i < i_min_max_values.size(); i++) {
          observers_[num_ops_id - 1].inputs_min_max_values[i][0] = (1 - c) * inputs_pre[i][0] + c * i_min_max_values[i][0];
          observers_[num_ops_id - 1].inputs_min_max_values[i][1] = (1 - c) * inputs_pre[i][1] + c * i_min_max_values[i][1];
        }
        for (auto j = 0; j < o_min_max_values.size(); j++) {
          observers_[num_ops_id - 1].outputs_min_max_values[j][0] = (1 - c) * outputs_pre[j][0] + c * o_min_max_values[j][0];
          observers_[num_ops_id - 1].outputs_min_max_values[j][1] = (1 - c) * outputs_pre[j][1] + c * o_min_max_values[j][1];
        }
      }
    }
  }

  inline void clear_indicators() {
    indicators_.clear();
  }

  inline void add_indicators() {
    num_ops_id = 0;
    indicators_.clear();
    // default used is s8, TODO: check inputs_dtype_uint8 and outputs_dtype_uint8 get the scales.
    for (auto i = 0; i < observers_.size(); i++) {
      std::vector<float> inputs_scale, outputs_scale;
      std::vector<std::vector<float>> inputs_values = observers_[i].inputs_min_max_values;
      std::vector<std::vector<float>> outputs_values = observers_[i].outputs_min_max_values;

      for (auto i = 0; i < inputs_values.size(); i++) {
        inputs_scale.push_back(127.5 / std::max(std::abs(inputs_values[i][0]), inputs_values[i][1]));
      }
      for (auto j = 0; j < outputs_values.size(); j++ ) {
        outputs_scale.push_back(127.5 / std::max(std::abs(outputs_values[j][0]), outputs_values[j][1]));
      }
      // zero_points not used now, zero_points = 0 for u8 and 128 for s8.
      //zero_point = 128;
      Indicator new_indicator(observers_[i].id, observers_[i].name, observers_[i].algorithm,
        observers_[i].weight_granularity, inputs_scale, outputs_scale, observers_[i].inputs_dtype_uint8,
        observers_[i].outputs_dtype_uint8, observers_[i].quantized);
      indicators_.push_back(new_indicator);
    }
    observers_.clear();
  }

  inline std::tuple<std::vector<std::vector<float>>, bool> get_indicator_scales(std::vector<bool> i_uint8_used, std::vector<bool> o_uint8_used) {
    std::vector<float> inputs_scale, outputs_scale;
    std::vector<bool> inputs_uint8_used, outputs_uint8_used;
    bool quantized_status;
    std::tie(inputs_uint8_used, outputs_uint8_used) = indicators_[num_ops_id].get_indicator_uint8_status();
    std::tie(inputs_scale, outputs_scale) = indicators_[num_ops_id].get_indicator_scales();
    quantized_status = indicators_[num_ops_id].get_indicator_quantized_status();
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
      indicators_[num_ops_id].set_indicator_scales(inputs_scale, outputs_scale);
      indicators_[num_ops_id].set_indicator_uint8_status(inputs_uint8_used, outputs_uint8_used);
    }
    num_ops_id++;
    // if whole workload has been run, reset the num_ops_id to zero.
    if (num_ops_id > indicators_.size() - 1) num_ops_id = 0;
    std::vector<std::vector<float>> input_output_scale = {inputs_scale, outputs_scale};
    return std::make_tuple(input_output_scale, quantized_status);
  }

  void set_indicators(std::vector<Indicator> indicators) { 
    indicators_ = indicators;
  }

  std::vector<Indicator> get_indicators() {
    return indicators_;
  }

  inline void calibration_reset() {
    num_ops_id = 0;
  }

private:
  AutoOptConfig() : auto_dnnl_(true), mix_bf16_fp32_(false), mix_int8_fp32_(false), num_ops_id(0),
    calibration_step_(false), observers_{}, indicators_{}, jit_fuse_(true), train_(false) {}
  ~AutoOptConfig() = default;
  AutoOptConfig(const AutoOptConfig&) = default;
  AutoOptConfig& operator=(const AutoOptConfig&) = default;

private:
  bool auto_dnnl_;
  bool jit_fuse_;
  bool mix_bf16_fp32_;
  bool train_;
  // int8
  bool mix_int8_fp32_;
  int64_t num_ops_id; // id number of call int8 path
  // the flag for one iteration of calibration step whether end or not
  bool calibration_step_;
  std::vector<Observer> observers_;
  std::vector<Indicator> indicators_;
};

} // namespace torch_ipex
