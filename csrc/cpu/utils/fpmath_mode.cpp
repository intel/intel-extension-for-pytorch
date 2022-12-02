#include "fpmath_mode.h"
#include <ideep.hpp>
#include <exception>
#include <iostream>
#include "csrc/cpu/ideep/IDeepConversions.h"

namespace torch_ipex {

dnnl_fpmath_mode_t fpmath_mode = []() {
  dnnl_fpmath_mode_t mode = dnnl_fpmath_mode_strict;
  static char* val = getenv("IPEX_FP32_MATH_MODE");
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

void setFP32MathModeCpu(FP32MathMode m) {
  dnnl_fpmath_mode_t mode = dnnl_fpmath_mode_strict;
  if (m == FP32MathMode::FP32) {
    mode = dnnl_fpmath_mode_strict;
  } else if (m == FP32MathMode::BF32) {
    mode = dnnl_fpmath_mode_bf16;
  }
  torch_ipex::fpmath_mode = mode;
}

FP32MathMode getFP32MathModeCpu() {
  dnnl_fpmath_mode_t mode = torch_ipex::fpmath_mode;
  switch (mode) {
    case dnnl_fpmath_mode_bf16:
      return FP32MathMode::BF32;
    default:
      return FP32MathMode::FP32;
  }
}

} // namespace torch_ipex
