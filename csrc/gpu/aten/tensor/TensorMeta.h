#pragma once

#include <ATen/ATen.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/strides.h>

namespace at {
namespace AtenIpexTypeXPU {

c10::optional<Tensor> set_strided(
    Tensor& output,
    IntArrayRef sizes,
    IntArrayRef strides,
    TensorOptions options);

c10::optional<Tensor> set_contiguous(
    Tensor& output,
    IntArrayRef sizes,
    TensorOptions options);

void set_contiguous_no_create(
    Tensor& output,
    IntArrayRef sizes,
    TensorOptions options);

} // namespace AtenIpexTypeXPU
} // namespace at