#pragma once
#include <ATen/Config.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include "ScalarType.h"

namespace at {
namespace AtenIpexTypeXPU {

template <typename T>
struct AccumulateType {};

template <>
struct AccumulateType<bool> {
  using type = bool;
};

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

template <>
struct AccumulateType<c10::complex<float>> {
  using type = c10::complex<float>;
};

template <>
struct AccumulateType<c10::complex<double>> {
  using type = c10::complex<double>;
};

template <typename T>
using acc_type = typename AccumulateType<T>::type;

// This function always return accumulator type for dpcpp
TORCH_API static inline c10::ScalarType toAccumulateType(c10::ScalarType type) {
  switch (type) {
#define DEFINE_CASE(scalar_t, TypeNum) \
  case ScalarType::TypeNum:            \
    return CppTypeToScalarType<at::AtenIpexTypeXPU::acc_type<scalar_t>>::value;

    IPEX_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_CASE)
#undef DEFINE_CASE

    default:
      TORCH_INTERNAL_ASSERT(false, "Unrecognized ScalarType: ", type);
  }
}

} // namespace AtenIpexTypeXPU
} // namespace at
