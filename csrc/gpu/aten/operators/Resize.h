#pragma once

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>

namespace at {
namespace AtenIpexTypeXPU {

bool resize_output_check(const Tensor& output, IntArrayRef shape);

bool resize_output(const Tensor& output, IntArrayRef shape);

} // namespace AtenIpexTypeXPU
} // namespace at
