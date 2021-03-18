#pragma once

#include "quantization/Observer.hpp"

namespace torch_ipex {

class AutoOptConfig {
public:
  static AutoOptConfig& singleton() {
    static AutoOptConfig auto_opt_conf;
    return auto_opt_conf;
  }

public:
  // int8
  inline void set_int8_calibration(bool value) {
    calibration_step_ = value;
  }
  inline bool get_int8_calibration() {
    return calibration_step_;
  }

private:
  AutoOptConfig() :  calibration_step_(false) {}

  ~AutoOptConfig() = default;
  AutoOptConfig(const AutoOptConfig&) = default;
  AutoOptConfig& operator=(const AutoOptConfig&) = default;

private:
  // the flag for one iteration of calibration step whether end or not
  bool calibration_step_;
};

} // namespace torch_ipex
