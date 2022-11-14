#pragma once

#include <tensor/Context.h>
#include <tensor/OpaqueTensorFactories.h>

namespace at {
namespace AtenIpexTypeXPU {

TensorImpl* resize_impl(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool device_guard = true);

bool check_has_opaque_and_no_padding(std::vector<at::Tensor> tlist);

Tensor share_storage_and_set_strided_as(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset_);

} // namespace AtenIpexTypeXPU
} // namespace at
