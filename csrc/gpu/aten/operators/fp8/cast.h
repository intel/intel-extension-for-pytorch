#include <ATen/ATen.h>
#include "fp8_utils.h"

namespace at {
namespace AtenIpexTypeXPU {

at::ScalarType convert_to_dtype(int64_t format);

void fp8_quantize_op(
    const Tensor& input,
    Tensor& input_fp8,
    int fp8_format,
    float* amax,
    const float* scale,
    float* scale_inv);

void fp8_dequantize_op(
    const Tensor& input,
    Tensor& input_c,
    int64_t itype,
    const float* scale_inv);

} // namespace AtenIpexTypeXPU
} // namespace at
