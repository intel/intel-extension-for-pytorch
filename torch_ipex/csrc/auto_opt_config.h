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
    std::vector<float> input_min_max_values, std::vector<float> output_min_max_values) {
    num_ops_id++;
    if (observers_.size() < num_ops_id) {
      // this path is that user not set int8 op's configure, using default configures
      Observer new_observer = {num_ops_id - 1, op_name, input_min_max_values, output_min_max_values};
      observers_.push_back(new_observer);
    } else {
      // user has set configure or have run one interation
      auto input_pre = observers_[num_ops_id - 1].Input_min_max_values;
      auto output_pre = observers_[num_ops_id - 1].Output_min_max_values;
      if (observers_[num_ops_id - 1].Algorithm == "min_max") {
        observers_[num_ops_id - 1].Input_min_max_values[0] = std::min(input_pre[0], input_min_max_values[0]);
        observers_[num_ops_id - 1].Input_min_max_values[1] = std::max(input_pre[1], input_min_max_values[1]);
        observers_[num_ops_id - 1].Output_min_max_values[0] = std::min(output_pre[0], output_min_max_values[0]);
        observers_[num_ops_id - 1].Output_min_max_values[1] = std::max(output_pre[1], output_min_max_values[1]);
      } else if(observers_[num_ops_id -1].Algorithm == "moving_averager_min_max"){
        auto c = observers_[num_ops_id - 1].Averaging_constant;
        observers_[num_ops_id - 1].Input_min_max_values[0] = (1 - c) * input_pre[0] + c * input_min_max_values[0];
        observers_[num_ops_id - 1].Input_min_max_values[1] = (1 - c) * input_pre[1] + c * input_min_max_values[1];
        observers_[num_ops_id - 1].Output_min_max_values[0] = (1 - c) * output_pre[0] + c * output_min_max_values[0];
        observers_[num_ops_id - 1].Output_min_max_values[1] = (1 - c) * output_pre[1] + c * output_min_max_values[1];
      }
    }
  }

  /*
  inline void print_observer() {
    for (auto i = 0; i< observers_.size(); i++) {
      for (auto j = 0; j < observers_[i].max_values.size(); j++)
        std::cout<<observers_[i].max_values[j]<<std::endl;
    }
  }
*/
  inline void print_indicator() {
    for (auto i = 0; i< indicators_.size(); i++) {
      auto scales = indicators_[i].get_indicator_scales();
      for (auto j = 0; j< scales.size(); j++)
          std::cout<<scales[j]<<std::endl;
    }
  }
 
  inline void add_indicators() {
    num_ops_id = 0;
    // default used is s8
    for (auto i = 0; i < observers_.size(); i++) {
      std::vector<float> scales;
      std::vector<float> input_values = observers_[i].Input_min_max_values;
      std::vector<float> output_values = observers_[i].Output_min_max_values;

      scales.push_back(127.5 / std::max(std::abs(input_values[0]), input_values[1]));
      scales.push_back(127.5 / std::max(std::abs(output_values[0]), output_values[1]));
      // zero_points not used now, zero_points = 0 for u8 and 128 for s8.
      //zero_point = 128;
      Indicator new_indicator(observers_[i].Id, observers_[i].Name, observers_[i].Algorithm,
        observers_[i].Weight_granularity, scales, {observers_[i].Input_dtype_uint8, observers_[i].Output_dtype_uint8},
        observers_[i].Quantized);
      indicators_.push_back(new_indicator);
    }
    observers_.clear();
  }

  inline std::tuple<std::vector<float>, bool> get_indicator_scales(std::vector<bool> uint8_used) {
    if (num_ops_id > indicators_.size() - 1) num_ops_id = 0;

    auto indicator_uint8_used = indicators_[num_ops_id].get_indicator_uint8_status();
    std::vector<float> indicator_scales;
    bool quantized_status;
    indicator_scales = indicators_[num_ops_id].get_indicator_scales();
    quantized_status = indicators_[num_ops_id].get_indicator_quantized_status();
    bool scale_update = false;
    for (auto i = 0; i < uint8_used.size(); i++) {
      if (!indicator_uint8_used[i] && uint8_used[i]) {
        // update zero_point and scales
        indicator_scales[i] /= 127.5;
        indicator_scales[i] *= 255.5;
        scale_update = true;
      } else if (indicator_uint8_used[i] && !uint8_used[i]) {
        // update zero_point and scales
        indicator_scales[i] /= 255.5;
        indicator_scales[i] *= 127.5;
        scale_update = true;
      }
    }
    if (scale_update) {
      indicators_[num_ops_id].set_indicator_scales(indicator_scales);
      indicators_[num_ops_id].set_indicator_uint8_status(uint8_used);
    }
    num_ops_id++;
    return std::make_tuple(indicator_scales, quantized_status);
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
  AutoOptConfig() : auto_dnnl_(true), mix_bf16_fp32_(false), num_ops_id(0), mix_int8_fp32_(false),
    calibration_step_(false), observers_{}, indicators_{}, jit_fuse_(true) {}
  ~AutoOptConfig() = default;
  AutoOptConfig(const AutoOptConfig&) = default;
  AutoOptConfig& operator=(const AutoOptConfig&) = default;

private:
  bool auto_dnnl_;
  bool jit_fuse_;
  bool mix_bf16_fp32_;
  // int8
  bool mix_int8_fp32_;
  int64_t num_ops_id; // id number of call int8 path
  // the flag for one iteration of calibration step whether end or not 
  bool calibration_step_;
  std::vector<Observer> observers_;
  std::vector<Indicator> indicators_;
};

} // namespace torch_ipex
