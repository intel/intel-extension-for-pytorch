#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/util/Exception.h>

#include <core/Context.h>
#include <core/TensorImplUtils.h>
#include <oneDNN/oneDNN.h>
#include "comm/Numerics.h"

namespace at {
namespace AtenIpexTypeQuantizedXPU {

Tensor requantize(
    const Tensor& rtensor,
    const double& scale_out,
    const int64_t& zero_point_out);

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
