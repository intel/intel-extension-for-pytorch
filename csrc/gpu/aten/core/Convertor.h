#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>

using namespace at;

namespace xpu {
namespace dpcpp {
namespace detail {

at::Device getATenDeviceFromUSM(void* src, const DeviceIndex device_id);
} // namespace detail

// This convertor will:
// 1) take a pointer src allocated in USM, data type, and layout info and
// convert it to the ATen tensor for zero-copy.
// 2) take a tensor object and convert it to a pointer addressed in USM.
// 3) this convertor doesn't manage USM pointer src's lifetime. please take care
// it carefully by yourself.
Tensor fromUSM(
    void* src,
    const ScalarType stype,
    IntArrayRef shape,
    c10::optional<IntArrayRef> strides = c10::nullopt,
    const DeviceIndex device_id = -1);

void* toUSM(const Tensor& src);
} // namespace dpcpp
} // namespace xpu
