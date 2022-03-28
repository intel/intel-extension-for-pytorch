#include "fpmath_mode.h"
#include <exception>
#include <iostream>
#include "csrc/cpu/ideep/ideep.hpp"
namespace ideep {
dnnl_fpmath_mode_t fpmath_mode = []() {
  dnnl_fpmath_mode_t mode = dnnl_fpmath_mode_strict;
  static char* val = getenv("IPEX_FP32_LOW_PRECISION_MODE_CPU");
  if (val != NULL) {
    std::string fpmath_mode = val;
    if (!fpmath_mode.empty()) {
      if (fpmath_mode.compare("BF32") == 0) {
        mode = dnnl_fpmath_mode_bf16;
      }
    }
  }
  return mode;
}();
}
namespace torch_ipex {

void setFP32LowPrecisionModeCpu(IPEXLowPrecisionMode m) {
  dnnl_fpmath_mode_t mode;
  if (m == IPEXLowPrecisionMode::FP32) {
    mode = dnnl_fpmath_mode_strict;
  } else if (m == IPEXLowPrecisionMode::BF32) {
    mode = dnnl_fpmath_mode_bf16;
  }
  ideep::utils::set_fpmath_mode(mode);
}

IPEXLowPrecisionMode getFP32LowPrecisionModeCpu() {
  dnnl_fpmath_mode_t mode = ideep::utils::get_fpmath_mode();
  switch (mode) {
    case dnnl_fpmath_mode_bf16:
      return IPEXLowPrecisionMode::BF32;
    default:
      return IPEXLowPrecisionMode::FP32;
  }
}

} // namespace torch_ipex
