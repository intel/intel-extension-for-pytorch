#pragma once
#include <ATen/ATen.h>

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

 private:
  AutoOptConfig()
      : jit_fuse_(true),
        calibration_step_(false),
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
