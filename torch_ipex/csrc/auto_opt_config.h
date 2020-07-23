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

  inline void insert_or_updata_observer(float max_value) {
    num_ops_id++;
    if (observers_.size() < num_ops_id) {
      Observer new_observer = {num_ops_id - 1, max_value};
      observers_.push_back(new_observer);
    } else {
      observers_[num_ops_id -1].max_value = std::max(observers_[num_ops_id -1].max_value, max_value);
    }
  }

  inline void print_observer() {
    std::cout<<"print observer data"<<std::endl;
    for (auto i = 0; i< observers_.size(); i++) {
      std::cout<<observers_[i].max_value<<std::endl;
    }
  }

  inline void print_indictor() {
    std::cout<<"print indictor"<<std::endl;
    for (auto i = 0; i< indicators_.size(); i++) {
      std::cout<<indicators_[i].scale<<std::endl;
      std::cout<<indicators_[i].zero_point<<std::endl;
    }
  }

  inline void add_indictors() {
    num_ops_id = 0;
    // default used is s8
    for (auto i = 0; i < observers_.size(); i++) {
      float scale, zero_point;
      scale = 127.0 / observers_[i].max_value;
      // zero_points not used now, zero_points = 0 for u8 and 128 for s8.
      zero_point = 128;
      Indicator new_indicator = {observers_[i].id, scale, zero_point, /* uint8_used*/false};
      indicators_.push_back(new_indicator);
    }
  }

  inline float get_indictor_scale(bool uint8_used) {
    if (num_ops_id > indicators_.size() - 1) num_ops_id = 0;

    if (!indicators_[num_ops_id].uint8_used && uint8_used) {
      // update zero_point and scales
      indicators_[num_ops_id].zero_point = 0;
      indicators_[num_ops_id].scale /= 127.0;
      indicators_[num_ops_id].scale *= 255.0;
      indicators_[num_ops_id].uint8_used = true;
    } else if (indicators_[num_ops_id].uint8_used && !uint8_used) {
      // update zero_point and scales
      indicators_[num_ops_id].zero_point = 128;
      indicators_[num_ops_id].scale /= 255;
      indicators_[num_ops_id].scale *= 127;
    }
    num_ops_id++;
    return indicators_[num_ops_id - 1].scale;
  }

  inline void calibration_reset() {
    //calibration_reseted = true;
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
  bool calibration_step_;
  // the flag for one iteration of calibration step whether end or not
  std::vector<Observer> observers_;
  std::vector<Indicator> indicators_;
};

} // namespace torch_ipex
