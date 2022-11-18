#pragma once

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>

namespace at {
namespace AtenIpexTypeXPU {

const Tensor& resize_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> memory_format);

const Tensor& resize_as_(
    const Tensor& self,
    const Tensor& the_template,
    c10::optional<MemoryFormat> memory_format);

bool resize_output_check(const Tensor& output, IntArrayRef shape);

bool resize_output(const Tensor& output, IntArrayRef shape);

} // namespace AtenIpexTypeXPU
} // namespace at
