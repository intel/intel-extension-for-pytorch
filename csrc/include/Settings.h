#pragma once

#include "Macros.h"

namespace xpu {

enum IPEX_API XPU_BACKEND {
  GPU = 0,
  CPU = 1,
  AUTO = 2,
  XPU_BACKEND_MAX = AUTO
};
static const char* IPEX_API XPU_BACKEND_STR[]{"GPU", "CPU", "AUTO"};

enum IPEX_API FP32_MATH_MODE {
  FP32 = 0,
  TF32 = 1,
  BF32 = 2,
  FP32_MATH_MODE_MAX = BF32
};
static const char* IPEX_API FP32_MATH_MODE_STR[]{"FP32", "TF32", "BF32"};

IPEX_API XPU_BACKEND get_backend();

IPEX_API bool set_backend(XPU_BACKEND backend);

IPEX_API FP32_MATH_MODE get_fp32_math_mode();

IPEX_API bool set_fp32_math_mode(FP32_MATH_MODE mode);

} // namespace xpu
