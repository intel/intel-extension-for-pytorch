#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#pragma once
namespace at {
namespace AtenIpexTypeXPU {
std::vector<int64_t> pool_output_sizes(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding_l,
    IntArrayRef padding_r,
    IntArrayRef dilation,
    bool ceil_mode);
}
} // namespace at
