#pragma once

#include <type_traits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551257390
#endif

#define IS_FLOAT(scalar_t)                  \
  (std::is_same<scalar_t, float>::value ||  \
   std::is_same<scalar_t, double>::value || \
   std::is_same<scalar_t, at::Half>::value)

#define IS_INTEGRAL(scalar_t)                \
  (std::is_same<scalar_t, uint8_t>::value || \
   std::is_same<scalar_t, int8_t>::value ||  \
   std::is_same<scalar_t, int32_t>::value || \
   std::is_same<scalar_t, int64_t>::value || \
   std::is_same<scalar_t, int16_t>::value)

#define IS_BFLOAT16(scalar_t) (std::is_same<scalar_t, at::BFloat16>::value)

#define IS_HALF(scalar_t) (std::is_same<scalar_t, at::Half>::value)

#define IS_FLOAT32(scalar_t) (std::is_same<scalar_t, float>::value)

#define IS_INT(scalar_t) (std::is_same<scalar_t, int>::value)

#define IS_INT64(scalar_t) (std::is_same<scalar_t, int64_t>::value)

#define IS_BOOL(scalar_t) (std::is_same<scalar_t, bool>::value)
