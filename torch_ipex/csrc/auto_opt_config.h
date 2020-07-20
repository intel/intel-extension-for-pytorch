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

  inline void insert_or_updata_observer(std::vector<int64_t> input_sizes, int64_t channel_axis,
      std::vector<float> mins, std::vector<float> maxs) {
    num_ops_id++;
    if (observers_.size() < num_ops_id) {
      Observer new_observer = {num_ops_id - 1, input_sizes, channel_axis, mins, maxs};
      observers_.push_back(new_observer);
    } else {
      for (auto i = 0; i < mins.size(); i++) {
        observers_[num_ops_id -1].mins[i] = std::min(observers_[num_ops_id -1].mins[i], mins[i]);
        observers_[num_ops_id -1].maxs[i] = std::max(observers_[num_ops_id -1].maxs[i], maxs[i]);
      }
    }
  }

  inline void print_observer() {
    for (auto i = 0; i< observers_.size(); i++) {
       std::cout<<"print i"<<i<<std::endl;
       for (auto j = 0; j < observers_[i].mins.size(); j++) {
          std::cout<<observers_[i].mins[j]<<std::endl;
          std::cout<<observers_[i].maxs[j]<<std::endl;
       }
    }
  }

  inline void print_indictor() {
    for (auto i = 0; i< indicators_.size(); i++) {
       std::cout<<"print i"<<i<<std::endl;
       for (auto j = 0; j < indicators_[i].scales.size(); j++) {
          std::cout<<indicators_[i].scales[j]<<std::endl;
          std::cout<<indicators_[i].zero_points[j]<<std::endl;
       }
    }
  }

  inline void add_indictors() {
    num_ops_id = 0;
    auto Qmin = -128; // -128 for s8, and 0 for u8
    // default used is s8
    for (auto i = 0; i < observers_.size(); i++) {
      std::vector<float> scales, zero_points;
      for (auto j = 0; j < observers_[i].mins.size(); j++) {
        scales.push_back(127 / observers_[i].maxs[j]);
        // zero_points not used now, zero_points = 0 for u8 and 128 for s8.
        zero_points.push_back(128);
      }
      Indicator new_indicator = {observers_[i].id, observers_[i].input_sizes, observers_[i].channel_axis,
          scales, zero_points, /* uint8_used*/false};
      indicators_.push_back(new_indicator);
    }
  }

  inline std::tuple<std::vector<float>, std::vector<float>> get_indictor_scales(bool uint8_used) {
    if (num_ops_id > indicators_.size() - 1) num_ops_id = 0;

    if (!indicators_[num_ops_id].uint8_used && uint8_used) {
      // update zero_point and scales
      for (auto i = 0; i < indicators_[num_ops_id].zero_points.size(); i++) {
        indicators_[num_ops_id].zero_points[i] = 0;
        indicators_[num_ops_id].scales[i] /= 127.0;
        indicators_[num_ops_id].scales[i] *= 255.0;
      }
      indicators_[num_ops_id].uint8_used = true;
    } else if (indicators_[num_ops_id].uint8_used && !uint8_used) {
      // update zero_point and scales
      for (auto i = 0; i < indicators_[num_ops_id].zero_points.size(); i++) {
        indicators_[num_ops_id].zero_points[i] = 128;
        indicators_[num_ops_id].scales[i] /= 255;
        indicators_[num_ops_id].scales[i] *= 127;
      }
    }
    num_ops_id++;
    return std::make_tuple(indicators_[num_ops_id - 1].scales, indicators_[num_ops_id - 1].zero_points);
  }

  inline void calibration_reset() {
    //calibration_reseted = true;
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
  bool calibration_step_;
  // the flag for one iteration of calibration step whether end or not
  std::vector<Observer> observers_;
  std::vector<Indicator> indicators_;
};

} // namespace torch_ipex
