#pragma once

#include <ATen/ATen.h>
#include <ATen/quantized/Quantizer.h>

namespace at {
namespace AtenIpexTypeXPU {

Tensor new_qtensor(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer);

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

using namespace at::AtenIpexTypeXPU;

int64_t q_zero_point(const Tensor& self);

double q_scale(const Tensor& self);

QScheme qscheme(const Tensor& self);

Tensor q_per_channel_scales(const Tensor& self);

Tensor q_per_channel_zero_points(const Tensor& self);

int64_t q_per_channel_axis(const Tensor& self);

Tensor& set_(
    Tensor& self,
    Storage storage,
    int64_t storage_offset,
    IntArrayRef sizes,
    IntArrayRef strides);

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
