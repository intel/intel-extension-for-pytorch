#pragma once
#include <ATen/Config.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace at {
namespace AtenIpexTypeXPU {

template <typename T>
struct AccumulateType {};

template <>
struct AccumulateType<at::Half> {
  using type = float;
};
template <>
struct AccumulateType<at::BFloat16> {
  using type = float;
};
template <>
struct AccumulateType<float> {
  using type = float;
};
template <>
struct AccumulateType<double> {
  using type = double;
};
template <>
struct AccumulateType<int8_t> {
  using type = int64_t;
};
template <>
struct AccumulateType<uint8_t> {
  using type = int64_t;
};
template <>
struct AccumulateType<char> {
  using type = int64_t;
};
template <>
struct AccumulateType<int16_t> {
  using type = int64_t;
};
template <>
struct AccumulateType<int32_t> {
  using type = int64_t;
};
template <>
struct AccumulateType<int64_t> {
  using type = int64_t;
};

template <typename T>
using acc_type = typename AccumulateType<T>::type;

} // namespace AtenIpexTypeXPU
} // namespace at
