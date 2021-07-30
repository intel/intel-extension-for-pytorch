#pragma once
#include <ATen/ATen.h>

#include "quantization/Observer.hpp"

namespace torch_ipex {

class AutoOptConfig {
public:
  static AutoOptConfig& singleton() {
    static AutoOptConfig auto_opt_conf;
    return auto_opt_conf;
  }
  inline void set_jit_fuse(bool jit_fuse) {
    jit_fuse_ = jit_fuse;
  }

  inline bool get_jit_fuse() {
    return jit_fuse_;
  }

  // int8
  inline void set_int8_calibration(bool value) {
    calibration_step_ = value;
  }
  inline bool get_int8_calibration() {
    return calibration_step_;
  }
  inline void set_int8_qscheme(const int scheme) {
    switch (scheme) {
    case 0:
      qscheme_ = at::QScheme::PER_TENSOR_AFFINE;
      break;
    case 1:
      qscheme_ = at::QScheme::PER_CHANNEL_AFFINE;
      break;
    case 2:
      qscheme_ = at::QScheme::PER_TENSOR_SYMMETRIC;
      break;
    case 3:
      qscheme_ = at::QScheme::PER_CHANNEL_SYMMETRIC;
      break;
    case 4:
      qscheme_ = at::QScheme::PER_CHANNEL_AFFINE_FLOAT_QPARAMS;
      break;
    default:
      TORCH_CHECK(false, "Unrecognized qscheme: ", static_cast<int>(scheme));
    }
  }
  inline at::QScheme get_int8_qscheme() { return qscheme_; }

private:
  AutoOptConfig()
      : jit_fuse_(true), calibration_step_(false),
        qscheme_(at::QScheme::PER_TENSOR_AFFINE) {}

  ~AutoOptConfig() = default;
  AutoOptConfig(const AutoOptConfig&) = default;
  AutoOptConfig& operator=(const AutoOptConfig&) = default;

  bool jit_fuse_;
  // the flag for one iteration of calibration step whether end or not.
  bool calibration_step_;
  at::QScheme qscheme_;
};

} // namespace torch_ipex
