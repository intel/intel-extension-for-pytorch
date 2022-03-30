#pragma once

#include <ATen/ATen.h>

namespace torch_ipex {

enum IPEXLowPrecisionMode : int { FP32 = 0, BF32 = 1 };

void setFP32LowPrecisionModeCpu(
    IPEXLowPrecisionMode mode = IPEXLowPrecisionMode::BF32);

IPEXLowPrecisionMode getFP32LowPrecisionModeCpu();

} // namespace torch_ipex
