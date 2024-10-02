#pragma once

#include <ATen/Context.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>

namespace at::AtenIpexTypeXPU {
namespace detail {

TensorBase empty_xpu(
    IntArrayRef size,
    ScalarType dtype,
    c10::optional<Device> device_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  const auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_xpu());
  const c10::DeviceGuard device_guard(device);
  auto* allocator = c10::GetAllocator(kXPU);
  constexpr c10::DispatchKeySet xpu_dks(c10::DispatchKey::XPU);
  return at::detail::empty_generic(
      size, allocator, xpu_dks, dtype, memory_format_opt);
}

TensorBase empty_xpu(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  TORCH_CHECK(
      !pin_memory_opt.has_value() || !*pin_memory_opt,
      "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      layout_or_default(layout_opt) == Layout::Strided);

  const auto dtype = dtype_or_default(dtype_opt);
  return detail::empty_xpu(size, dtype, device_opt, memory_format_opt);
}

TensorBase empty_xpu(IntArrayRef size, const TensorOptions& options) {
  return detail::empty_xpu(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

TensorBase empty_strided_xpu(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    c10::optional<Device> device_opt) {
  const auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_xpu());
  const c10::DeviceGuard device_guard(device);
  auto* allocator = c10::GetAllocator(kXPU);
  constexpr c10::DispatchKeySet xpu_dks(c10::DispatchKey::XPU);
  return at::detail::empty_strided_generic(
      size, stride, allocator, xpu_dks, dtype);
}

TensorBase empty_strided_xpu(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  TORCH_CHECK(
      !pin_memory_opt.has_value() || !*pin_memory_opt,
      "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      layout_or_default(layout_opt) == Layout::Strided);

  const auto dtype = dtype_or_default(dtype_opt);
  return detail::empty_strided_xpu(size, stride, dtype, device_opt);
}

TensorBase empty_strided_xpu(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options) {
  return detail::empty_strided_xpu(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

} // namespace detail

inline Tensor create_out(
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options) {
  if (strides.empty()) {
    return detail::empty_xpu(sizes, options);
  } else {
    return detail::empty_strided_xpu(sizes, strides, options);
  }
}

} // namespace at::AtenIpexTypeXPU
