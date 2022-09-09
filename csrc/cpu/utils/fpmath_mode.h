#pragma once

#include <ATen/ATen.h>

namespace torch_ipex {

enum FP32MathMode : int { FP32 = 0, TF32 = 1, BF32 = 2 };

void setFP32MathModeCpu(FP32MathMode mode = FP32MathMode::FP32);

FP32MathMode getFP32MathModeCpu();

} // namespace torch_ipex
