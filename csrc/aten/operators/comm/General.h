#pragma once

#include <type_traits>

// Note: Please consider using Numerics<T>::pi() instead.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551257390
#endif

#define IS_BFLOAT16(scalar_t) (std::is_same<scalar_t, at::BFloat16>::value)

#define IS_HALF(scalar_t) (std::is_same<scalar_t, at::Half>::value)

#define IS_FLOAT32(scalar_t) (std::is_same<scalar_t, float>::value)

#define IS_DOUBLE(scalar_t) (std::is_same<scalar_t, double>::value)

#define IS_INT8(scalar_t) (std::is_same<scalar_t, int8_t>::value)

#define IS_UINT8(scalar_t) (std::is_same<scalar_t, uint8_t>::value)

#define IS_INT16(scalar_t) (std::is_same<scalar_t, int16_t>::value)

#define IS_INT(scalar_t) (std::is_same<scalar_t, int>::value)

#define IS_INT64(scalar_t) (std::is_same<scalar_t, int64_t>::value)

#define IS_BOOL(scalar_t) (std::is_same<scalar_t, bool>::value)

#define IS_COMPLEX_FLOAT(scalar_t) \
  (std::is_same<scalar_t, c10::complex<float>>::value)

#define IS_COMPLEX_DOUBLE(scalar_t) \
  (std::is_same<scalar_t, c10::complex<double>>::value)

#define IS_FLOAT(scalar_t)                                                 \
  (IS_DOUBLE(scalar_t) || IS_FLOAT32(scalar_t) || IS_BFLOAT16(scalar_t) || \
   IS_HALF(scalar_t))

#define IS_INTEGRAL(scalar_t)                                     \
  (IS_UINT8(scalar_t) || IS_INT8(scalar_t) || IS_INT(scalar_t) || \
   IS_INT64(scalar_t) || IS_INT16(scalar_t))

#define IS_COMPLEX(scalar_t) \
  (IS_COMPLEX_FLOAT(scalar_t) || IS_COMPLEX_DOUBLE(scalar_t))

struct NullType {
  using value_type = NullType;
  template <typename T>
  inline NullType& operator=(const T&) {
    return *this;
  }
  inline bool operator==(const NullType&) {
    return true;
  }
  inline bool operator!=(const NullType&) {
    return false;
  }
};
