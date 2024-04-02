#include <Macros.h>
#include <torch/csrc/jit/api/module.h>

namespace torch_ipex {

enum IPEX_API FP32MathMode : int { FP32 = 0, TF32 = 1, BF32 = 2 };

IPEX_API void setFP32MathModeCpu(FP32MathMode mode = FP32MathMode::FP32);

IPEX_API FP32MathMode getFP32MathModeCpu();

} // namespace torch_ipex
