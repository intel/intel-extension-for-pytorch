#pragma once

#include <ATen/ATen.h>
#include <ATen/quantized/Quantizer.h>

namespace at {
namespace AtenIpexTypeXPU {

Tensor new_qtensor(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer);

Tensor as_strided_quantized_dpcpp(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride);

Tensor expand_as_quantized_dpcpp(const Tensor& self, const Tensor& other);

} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {

using namespace at::AtenIpexTypeXPU;

int64_t q_zero_point(const Tensor& self);

double q_scale(const Tensor& self);

QScheme qscheme(const Tensor& self);

Tensor q_per_channel_scales(const Tensor& self);

Tensor q_per_channel_zero_points(const Tensor& self);

int64_t q_per_channel_axis(const Tensor& self);

Tensor& set_quantizer_(Tensor& self, ConstQuantizerPtr quantizer);

Tensor& set_(
    Tensor& self,
    Storage storage,
    int64_t storage_offset,
    IntArrayRef sizes,
    IntArrayRef strides);

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
