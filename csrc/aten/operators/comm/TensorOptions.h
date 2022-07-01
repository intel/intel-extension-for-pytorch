#pragma once

#include <utils/DPCPP.h>

namespace at {
namespace AtenIpexTypeXPU {

template <typename T>
DPCPP_DEVICE static inline TensorOptions map_options() {
  if (std::is_same<T, uint8_t>::value)
    return at::TensorOptions().dtype(kByte).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, int8_t>::value)
    return at::TensorOptions().dtype(kChar).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, int16_t>::value)
    return at::TensorOptions().dtype(kShort).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, int32_t>::value)
    return at::TensorOptions().dtype(kInt).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, int64_t>::value)
    return at::TensorOptions().dtype(kLong).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, float>::value)
    return at::TensorOptions().dtype(kFloat).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, double>::value)
    return at::TensorOptions().dtype(kDouble).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, at::Half>::value)
    return at::TensorOptions().dtype(kHalf).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, at::BFloat16>::value)
    return at::TensorOptions().dtype(kBFloat16).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, bool>::value)
    return at::TensorOptions().dtype(kBool).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, c10::complex<float>>::value)
    return at::TensorOptions()
        .dtype(kComplexFloat)
        .device(kXPU)
        .memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same<T, c10::complex<double>>::value)
    return at::TensorOptions()
        .dtype(kComplexDouble)
        .device(kXPU)
        .memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else {
    AT_ERROR("PSTLFunctions: data type cannot be mapped to tensor's dtype.");
  }
  return at::TensorOptions();
}

} // namespace AtenIpexTypeXPU
} // namespace at