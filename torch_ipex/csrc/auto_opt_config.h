#pragma once

#include "csrc/quantization/Observer.h"

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
  
  inline bool set_train(bool value) {
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

  inline void insert_or_updata_observer(std::string op_name, std::vector<float> max_values) {
    num_ops_id++;
    if (observers_.size() < num_ops_id) {
      //Operator op = {num_ops_id - 1, op_n};
      Observer new_observer = {num_ops_id - 1, op_name, max_values};
      observers_.push_back(new_observer);
    } else {
      for (auto i = 0; i < max_values.size(); i++)
        observers_[num_ops_id -1].max_values[i] = std::max(observers_[num_ops_id -1].max_values[i], max_values[i]);
    }
  }

  inline void print_observer() {
    for (auto i = 0; i< observers_.size(); i++) {
      for (auto j = 0; j < observers_[i].max_values.size(); j++)
        std::cout<<observers_[i].max_values[j]<<std::endl;
    }
  }

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
      std::vector<bool> uint8_used;
      for (auto j = 0; j < observers_[i].max_values.size(); j++) {
        scales.push_back(127.5 / observers_[i].max_values[j]);
        uint8_used.push_back(false);
      }
        // zero_points not used now, zero_points = 0 for u8 and 128 for s8.
        //zero_point = 128;
      Indicator new_indicator(observers_[i].Id, observers_[i].Name, scales, uint8_used, true);
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

  void save_indicators_file(const std::string& file_name) {
    std::ofstream out(file_name);
    auto indicators_number = indicators_.size();
    out << indicators_number << "\n"<<std::endl;;
    for (auto i = 0; i < indicators_number; i++) {
      out << indicators_[i];
    }
    out.close();
  }

  void load_indicators_file(const std::string& file_name) {
    std::ifstream in(file_name);
    int64_t indicators_number;
    in >> indicators_number;
    Indicator temp;
    indicators_.clear();
    for (auto i = 0; i < indicators_number; i++) {
      in >> temp;
      indicators_.push_back(temp);
    }
    in.close();
  }

  inline void calibration_reset() {
    num_ops_id = 0;
  }

private:
  AutoOptConfig() : auto_dnnl_(true), mix_bf16_fp32_(false), num_ops_id(0), mix_int8_fp32_(false),
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
