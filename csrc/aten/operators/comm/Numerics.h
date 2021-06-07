#pragma once

#include <ATen/ATen.h>

#include <utils/DPCPP.h>
#include "General.h"
#include "NumericLimits.h"

template <typename T>
struct Numerics {};

template <typename T>
static inline T powi(T a, T b) {
  T result = 1;
  while (b) {
    if (b & 1) {
      result *= a;
    }
    b /= 2;
    a *= a;
  }
  return result;
}

template<typename T>
static inline T sgni(T z) {
  if (z == (T)(0)) {
    return (T)(0);
  } else {
    return z / abs(z);
  }
}

template <>
struct Numerics<uint8_t> {
  static inline uint8_t lower_bound() {
    return at::numeric_limits<uint8_t>::lower_bound();
  }
  static inline uint8_t upper_bound() {
    return at::numeric_limits<uint8_t>::upper_bound();
  }

  static inline bool lt(uint8_t a, uint8_t b) {
    return a < b;
  }
  static inline bool le(uint8_t a, uint8_t b) {
    return a <= b;
  }
  static inline bool gt(uint8_t a, uint8_t b) {
    return a > b;
  }
  static inline bool ge(uint8_t a, uint8_t b) {
    return a >= b;
  }
  static inline bool eq(uint8_t a, uint8_t b) {
    return a == b;
  }
  static inline bool ne(uint8_t a, uint8_t b) {
    return a != b;
  }

  static inline uint8_t neg(uint8_t a) {
    return -a;
  }
  static inline uint8_t abs(uint8_t a) {
    return a;
  }
  static inline uint8_t add(uint8_t a, uint8_t b) {
    return a + b;
  }
  static inline uint8_t mul(uint8_t a, uint8_t b) {
    return a * b;
  }
  static inline uint8_t sub(uint8_t a, uint8_t b) {
    return a - b;
  }
  static inline uint8_t div(uint8_t a, uint8_t b) {
    return a / b;
  }
  static inline uint8_t pow(uint8_t a, uint8_t b) {
    return powi<uint8_t>(a, b);
  }
  static inline bool isnan(uint8_t a) {
    return false;
  }
  static inline bool isinf(uint8_t a) {
    return false;
  }
};

template <>
struct Numerics<bool> {
  static inline bool lower_bound() {
    return at::numeric_limits<bool>::lower_bound();
  }
  static inline bool upper_bound() {
    return at::numeric_limits<bool>::upper_bound();
  }

  static inline bool lt(bool a, bool b) {
    return a < b;
  }
  static inline bool le(bool a, bool b) {
    return a <= b;
  }
  static inline bool gt(bool a, bool b) {
    return a > b;
  }
  static inline bool ge(bool a, bool b) {
    return a >= b;
  }
  static inline bool eq(bool a, bool b) {
    return a == b;
  }
  static inline bool ne(bool a, bool b) {
    return a != b;
  }
  static inline bool add(bool a, bool b) {
    return a + b;
  }
  static inline bool mul(bool a, bool b) {
    return a && b;
  }
  static inline bool sub(bool a, bool b) {
    return a - b;
  }
  static inline bool div(bool a, bool b) {
    return a / b;
  }
  static inline bool abs(bool a) {
    return a;
  }
  static inline bool isnan(bool a) {
    return false;
  }
  static inline bool isinf(bool a) {
    return false;
  }
};

template <>
struct Numerics<int8_t> {
  static inline int8_t lower_bound() {
    return at::numeric_limits<int8_t>::lower_bound();
  }
  static inline int8_t upper_bound() {
    return at::numeric_limits<int8_t>::upper_bound();
  }

  static inline bool lt(int8_t a, int8_t b) {
    return a < b;
  }
  static inline bool le(int8_t a, int8_t b) {
    return a <= b;
  }
  static inline bool gt(int8_t a, int8_t b) {
    return a > b;
  }
  static inline bool ge(int8_t a, int8_t b) {
    return a >= b;
  }
  static inline bool eq(int8_t a, int8_t b) {
    return a == b;
  }
  static inline bool ne(int8_t a, int8_t b) {
    return a != b;
  }

  static inline int8_t neg(int8_t a) {
    return -a;
  }
  static inline int8_t add(int8_t a, int8_t b) {
    return a + b;
  }
  static inline int8_t mul(int8_t a, int8_t b) {
    return a * b;
  }
  static inline int8_t sub(int8_t a, int8_t b) {
    return a - b;
  }
  static inline int8_t div(int8_t a, int8_t b) {
    return a / b;
  }
  static inline int8_t abs(int8_t a) {
    return DPCPP::abs((int)a);
  }
  static inline int8_t pow(int8_t a, int8_t b) {
    return powi<int8_t>(a, b);
  }
  static inline bool isnan(int8_t a) {
    return false;
  }
  static inline bool isinf(int8_t a) {
    return false;
  }
};

template <>
struct Numerics<int16_t> {
  static inline int16_t lower_bound() {
    return at::numeric_limits<int16_t>::lower_bound();
  }
  static inline int16_t upper_bound() {
    return at::numeric_limits<int16_t>::upper_bound();
  }

  static inline bool lt(int16_t a, int16_t b) {
    return a < b;
  }
  static inline bool le(int16_t a, int16_t b) {
    return a <= b;
  }
  static inline bool gt(int16_t a, int16_t b) {
    return a > b;
  }
  static inline bool ge(int16_t a, int16_t b) {
    return a >= b;
  }
  static inline bool eq(int16_t a, int16_t b) {
    return a == b;
  }
  static inline bool ne(int16_t a, int16_t b) {
    return a != b;
  }

  static inline int16_t neg(int16_t a) {
    return -a;
  }
  static inline int16_t add(int16_t a, int16_t b) {
    return a + b;
  }
  static inline int16_t mul(int16_t a, int16_t b) {
    return a * b;
  }
  static inline int16_t sub(int16_t a, int16_t b) {
    return a - b;
  }
  static inline int16_t div(int16_t a, int16_t b) {
    return a / b;
  }
  static inline int16_t abs(int16_t a) {
    return DPCPP::abs(a);
  }
  static inline int16_t pow(int16_t a, int16_t b) {
    return powi<int16_t>(a, b);
  }

  static inline bool isnan(int16_t a) {
    return false;
  }

  static inline int16_t sgn(int16_t a) {
    return sgni<int16_t>(a);
  }
};

template <>
struct Numerics<int32_t> {
  static inline int32_t lower_bound() {
    return at::numeric_limits<int32_t>::lower_bound();
  }
  static inline int32_t upper_bound() {
    return at::numeric_limits<int32_t>::upper_bound();
  }

  static inline bool lt(int32_t a, int32_t b) {
    return a < b;
  }
  static inline bool le(int32_t a, int32_t b) {
    return a <= b;
  }
  static inline bool gt(int32_t a, int32_t b) {
    return a > b;
  }
  static inline bool ge(int32_t a, int32_t b) {
    return a >= b;
  }
  static inline bool eq(int32_t a, int32_t b) {
    return a == b;
  }
  static inline bool ne(int32_t a, int32_t b) {
    return a != b;
  }

  static inline int32_t neg(int32_t a) {
    return -a;
  }
  static inline int32_t add(int32_t a, int32_t b) {
    return a + b;
  }
  static inline int32_t mul(int32_t a, int32_t b) {
    return a * b;
  }
  static inline int32_t sub(int32_t a, int32_t b) {
    return a - b;
  }
  static inline int32_t div(int32_t a, int32_t b) {
    return a / b;
  }
  static inline int32_t abs(int32_t a) {
    return DPCPP::abs(a);
  }
  static inline int32_t pow(int32_t a, int32_t b) {
    return powi<int32_t>(a, b);
  }

  static inline bool isnan(int32_t a) {
    return false;
  }

  static inline int32_t sgn(int32_t a) {
    return sgni<int32_t>(a);
  }
};

template <>
struct Numerics<int64_t> {
  static inline int64_t lower_bound() {
    return at::numeric_limits<int64_t>::lower_bound();
  }
  static inline int64_t upper_bound() {
    return at::numeric_limits<int64_t>::upper_bound();
  }

  static inline bool lt(int64_t a, int64_t b) {
    return a < b;
  }
  static inline bool le(int64_t a, int64_t b) {
    return a <= b;
  }
  static inline bool gt(int64_t a, int64_t b) {
    return a > b;
  }
  static inline bool ge(int64_t a, int64_t b) {
    return a >= b;
  }
  static inline bool eq(int64_t a, int64_t b) {
    return a == b;
  }
  static inline bool ne(int64_t a, int64_t b) {
    return a != b;
  }

  static inline int64_t neg(int64_t a) {
    return -a;
  }
  static inline int64_t add(int64_t a, int64_t b) {
    return a + b;
  }
  static inline int64_t mul(int64_t a, int64_t b) {
    return a * b;
  }
  static inline int64_t sub(int64_t a, int64_t b) {
    return a - b;
  }
  static inline int64_t div(int64_t a, int64_t b) {
    return a / b;
  }
  static inline int64_t abs(int64_t a) {
    return DPCPP::abs(a);
  }
  static inline int64_t pow(int64_t a, int64_t b) {
    return powi<int64_t>(a, b);
  }

  static inline bool isnan(int64_t a) {
    return false;
  }

  static inline int64_t sgn(int64_t a) {
    return sgni<int64_t>(a);
  }
};

template <>
struct Numerics<at::Half> {
  static inline at::Half lower_bound() {
    return at::numeric_limits<at::Half>::lower_bound();
  }
  static inline at::Half upper_bound() {
    return at::numeric_limits<at::Half>::upper_bound();
  }

  static inline bool lt(at::Half a, at::Half b) {
    return a < b;
  }
  static inline bool le(at::Half a, at::Half b) {
    return a <= b;
  }
  static inline bool gt(at::Half a, at::Half b) {
    return a > b;
  }
  static inline bool ge(at::Half a, at::Half b) {
    return a >= b;
  }
  static inline bool eq(at::Half a, at::Half b) {
    return a == b;
  }
  static inline bool ne(at::Half a, at::Half b) {
    return a != b;
  }

  static inline at::Half exp(at::Half a) {
    return DPCPP::exp(float(a));
  }
  static inline at::Half exp10(at::Half a) {
    return DPCPP::exp10(float(a));
  }
  static inline at::Half log(at::Half a) {
    return DPCPP::log(float(a));
  }
  static inline at::Half log10(at::Half a) {
    return DPCPP::log10(float(a));
  }
  static inline at::Half log1p(at::Half a) {
    return DPCPP::log1p(float(a));
  }
  static inline at::Half log2(at::Half a) {
    return DPCPP::log2(float(a));
  }
  static inline at::Half expm1(at::Half a) {
    return DPCPP::expm1(float(a));
  }

  static inline at::Half neg(at::Half a) {
    return -a;
  }
  static inline at::Half sin(at::Half a) {
    return DPCPP::sin(float(a));
  }
  static inline at::Half cos(at::Half a) {
    return DPCPP::cos(float(a));
  }
  static inline at::Half sqrt(at::Half a) {
    return DPCPP::sqrt(float(a));
  }
  static inline at::Half rsqrt(at::Half a) {
    return DPCPP::rsqrt(float(a));
  }
  static inline at::Half ceil(at::Half a) {
    return DPCPP::ceil(float(a));
  }
  static inline at::Half floor(at::Half a) {
    return DPCPP::floor(float(a));
  }
  static inline at::Half trunc(at::Half a) {
    return DPCPP::trunc(float(a));
  }
  static inline at::Half acos(at::Half a) {
    return DPCPP::acos(float(a));
  }
  static inline at::Half cosh(at::Half a) {
    return DPCPP::cosh(float(a));
  }
  static inline at::Half asin(at::Half a) {
    return DPCPP::asin(float(a));
  }
  static inline at::Half sinh(at::Half a) {
    return DPCPP::sinh(float(a));
  }
  static inline at::Half tan(at::Half a) {
    return DPCPP::tan(float(a));
  }
  static inline at::Half atan(at::Half a) {
    return DPCPP::atan(float(a));
  }
  static inline at::Half tanh(float a) {
    return DPCPP::tanh(float(a));
  }
  static inline at::Half erf(float a) {
    return DPCPP::erf(float(a));
  }
  static inline at::Half erfc(float a) {
    return DPCPP::erfc(float(a));
  }
  static inline at::Half round(float a) {
    return DPCPP::round(float(a));
  }

  static inline at::Half frac(at::Half a) {
    return a - DPCPP::trunc(float(a));
  }
  static inline at::Half atan2(at::Half a, at::Half b) {
    return DPCPP::atan2(float(a), float(b));
  }
  static inline at::Half cinv(at::Half a) {
    return 1.0f / a;
  }
  static inline float min(at::Half a, at::Half b) {
    return DPCPP::fmin(float(a), float(b));
  }
  static inline at::Half add(at::Half a, at::Half b) {
    return a + b;
  }
  static inline at::Half div(at::Half a, at::Half b) {
    return a / b;
  }
  static inline at::Half mul(at::Half a, at::Half b) {
    return a * b;
  }
  static inline at::Half sub(at::Half a, at::Half b) {
    return a - b;
  }
  static inline at::Half pow(at::Half a, at::Half b) {
    return DPCPP::pow(float(a), float(b));
  }

  static inline at::Half max(at::Half a, at::Half b) {
    return DPCPP::fmax(float(a), float(b));
  }

  static inline at::Half abs(at::Half a) {
    return DPCPP::fabs(float(a));
  }
  static inline at::Half fabs(at::Half a) {
    return DPCPP::fabs(float(a));
  }
  static inline bool isnan(at::Half a) {
    return DPCPP::isnan((float)a);
  }

  static inline at::Half sgn(at::Half a) {
    return sgni<at::Half>(a);
  }
};

template <>
struct Numerics<at::BFloat16> {
  static inline at::BFloat16 lower_bound() {
    return at::numeric_limits<at::BFloat16>::lower_bound();
  }
  static inline at::BFloat16 upper_bound() {
    return at::numeric_limits<at::BFloat16>::upper_bound();
  }

  static inline bool lt(at::BFloat16 a, at::BFloat16 b) {
    return a < b;
  }
  static inline bool le(at::BFloat16 a, at::BFloat16 b) {
    return a <= b;
  }
  static inline bool gt(at::BFloat16 a, at::BFloat16 b) {
    return a > b;
  }
  static inline bool ge(at::BFloat16 a, at::BFloat16 b) {
    return a >= b;
  }
  static inline bool eq(at::BFloat16 a, at::BFloat16 b) {
    return a == b;
  }
  static inline bool ne(at::BFloat16 a, at::BFloat16 b) {
    return a != b;
  }

  static inline at::BFloat16 exp(at::BFloat16 a) {
    return DPCPP::exp(float(a));
  }
  static inline at::BFloat16 exp10(at::BFloat16 a) {
    return DPCPP::exp10(float(a));
  }
  static inline at::BFloat16 log(at::BFloat16 a) {
    return DPCPP::log(float(a));
  }
  static inline at::BFloat16 log10(at::BFloat16 a) {
    return DPCPP::log10(float(a));
  }
  static inline at::BFloat16 log1p(at::BFloat16 a) {
    return DPCPP::log1p(float(a));
  }
  static inline at::BFloat16 log2(at::BFloat16 a) {
    return DPCPP::log2(float(a));
  }
  static inline at::BFloat16 expm1(at::BFloat16 a) {
    return DPCPP::expm1(float(a));
  }

  static inline at::BFloat16 neg(at::BFloat16 a) {
    return -a;
  }
  static inline at::BFloat16 sin(at::BFloat16 a) {
    return DPCPP::sin(float(a));
  }
  static inline at::BFloat16 cos(at::BFloat16 a) {
    return DPCPP::cos(float(a));
  }
  static inline at::BFloat16 sqrt(at::BFloat16 a) {
    return DPCPP::sqrt(float(a));
  }
  static inline at::BFloat16 rsqrt(at::BFloat16 a) {
    return DPCPP::rsqrt(float(a));
  }
  static inline at::BFloat16 ceil(at::BFloat16 a) {
    return DPCPP::ceil(float(a));
  }
  static inline at::BFloat16 floor(at::BFloat16 a) {
    return DPCPP::floor(float(a));
  }
  static inline at::BFloat16 trunc(at::BFloat16 a) {
    return DPCPP::trunc(float(a));
  }
  static inline at::BFloat16 acos(at::BFloat16 a) {
    return DPCPP::acos(float(a));
  }
  static inline at::BFloat16 cosh(at::BFloat16 a) {
    return DPCPP::cosh(float(a));
  }
  static inline at::BFloat16 asin(at::BFloat16 a) {
    return DPCPP::asin(float(a));
  }
  static inline at::BFloat16 sinh(at::BFloat16 a) {
    return DPCPP::sinh(float(a));
  }
  static inline at::BFloat16 tan(at::BFloat16 a) {
    return DPCPP::tan(float(a));
  }
  static inline at::BFloat16 atan(at::BFloat16 a) {
    return DPCPP::atan(float(a));
  }
  static inline at::BFloat16 tanh(float a) {
    return DPCPP::tanh(float(a));
  }
  static inline at::BFloat16 erf(float a) {
    return DPCPP::erf(float(a));
  }
  static inline at::BFloat16 erfc(float a) {
    return DPCPP::erfc(float(a));
  }
  static inline at::BFloat16 round(float a) {
    return DPCPP::round(float(a));
  }
  static inline float min(at::BFloat16 a, at::BFloat16 b) {
    return DPCPP::fmin(float(a), float(b));
  }
  static inline float max(at::BFloat16 a, at::BFloat16 b) {
    return DPCPP::fmax(float(a), float(b));
  }
  static inline at::BFloat16 frac(at::BFloat16 a) {
    return a - DPCPP::trunc(float(a));
  }
  static inline at::BFloat16 atan2(at::BFloat16 a, at::BFloat16 b) {
    return DPCPP::atan2(float(a), float(b));
  }
  static inline at::BFloat16 cinv(at::BFloat16 a) {
    return 1.0f / a;
  }

  static inline at::BFloat16 add(at::BFloat16 a, at::BFloat16 b) {
    return a + b;
  }
  static inline at::BFloat16 div(at::BFloat16 a, at::BFloat16 b) {
    return a / b;
  }
  static inline at::BFloat16 mul(at::BFloat16 a, at::BFloat16 b) {
    return a * b;
  }
  static inline at::BFloat16 sub(at::BFloat16 a, at::BFloat16 b) {
    return a - b;
  }

  static inline at::BFloat16 pow(at::BFloat16 a, at::BFloat16 b) {
    return DPCPP::pow(float(a), float(b));
  }
  static inline at::BFloat16 abs(at::BFloat16 a) {
    return DPCPP::fabs(float(a));
  }
  static inline at::BFloat16 fabs(at::BFloat16 a) {
    return DPCPP::fabs(float(a));
  }
  static inline bool isnan(at::BFloat16 a) {
    return DPCPP::isnan((float)a);
  }
  static inline at::BFloat16 sgn(at::BFloat16 a) {
    return sgni<at::BFloat16>(a);
  }
};

template <>
struct Numerics<float> {
  static inline float lower_bound() {
    return at::numeric_limits<float>::lower_bound();
  }
  static inline float upper_bound() {
    return at::numeric_limits<float>::upper_bound();
  }

  static inline bool lt(float a, float b) {
    return DPCPP::isless(a, b);
  }
  static inline bool le(float a, float b) {
    return DPCPP::islessequal(a, b);
  }
  static inline bool gt(float a, float b) {
    return DPCPP::isgreater(a, b);
  }
  static inline bool ge(float a, float b) {
    return DPCPP::isgreaterequal(a, b);
  }
  static inline bool eq(float a, float b) {
    return DPCPP::isequal(a, b);
  }
  static inline bool ne(float a, float b) {
    return DPCPP::isnotequal(a, b);
  }

  static inline float exp(float a) {
    return DPCPP::exp(a);
  }
  static inline float exp10(float a) {
    return DPCPP::exp10(a);
  }
  static inline float log(float a) {
    return DPCPP::log(a);
  }
  static inline float log10(float a) {
    return DPCPP::log10(a);
  }
  static inline float log1p(float a) {
    return DPCPP::log1p(a);
  }
  static inline float log2(float a) {
    return DPCPP::log2(a);
  }
  static inline float expm1(float a) {
    return DPCPP::expm1(a);
  }

  static inline float neg(float a) {
    return -a;
  }
  static inline float sin(float a) {
    return DPCPP::sin(a);
  }
  static inline float cos(float a) {
    return DPCPP::cos(a);
  }
  static inline float sqrt(float a) {
    return DPCPP::sqrt(a);
  }
  static inline float rsqrt(float a) {
    return DPCPP::rsqrt(a);
  }
  static inline float ceil(float a) {
    return DPCPP::ceil(a);
  }
  static inline float floor(float a) {
    return DPCPP::floor(a);
  }
  static inline float trunc(float a) {
    return DPCPP::trunc(a);
  }

  static inline float acos(float a) {
    return DPCPP::acos(a);
  }
  static inline float cosh(float a) {
    return DPCPP::cosh(a);
  }
  static inline float asin(float a) {
    return DPCPP::asin(a);
  }
  static inline float sinh(float a) {
    return DPCPP::sinh(a);
  }
  static inline float tan(float a) {
    return DPCPP::tan(a);
  }
  static inline float atan(float a) {
    return DPCPP::atan(a);
  }
  static inline float tanh(float a) {
    return DPCPP::tanh(a);
  }
  static inline float erf(float a) {
    return DPCPP::erf(a);
  }
  static inline float erfc(float a) {
    return DPCPP::erfc(a);
  }
  static inline float round(float a) {
    return DPCPP::round(a);
  }

  static inline float frac(float a) {
    return a - DPCPP::trunc(a);
  }
  static inline float cinv(float a) {
    return 1.0f / a;
  }
  static inline float add(float a, float b) {
    return a + b;
  }
  static inline float div(float a, float b) {
    return a / b;
  }
  static inline float mul(float a, float b) {
    return a * b;
  }
  static inline float sub(float a, float b) {
    return a - b;
  }
  static inline float pow(float a, float b) {
    return DPCPP::pow(a, b);
  }
  static inline float atan2(float a, float b) {
    return DPCPP::atan2(a, b);
  }
  static inline float min(float a, float b) {
    return DPCPP::fmin(a, b);
  }
  static inline float max(float a, float b) {
    return DPCPP::fmax(a, b);
  }
  static inline float abs(float a) {
    return DPCPP::fabs(a);
  }
  static inline float fabs(float a) {
    return DPCPP::fabs(a);
  }
  static inline bool isnan(float a) {
    return DPCPP::isnan(a);
  }
  static inline float sgn(float a) {
    return sgni<float>(a);
  }
};

template <>
struct Numerics<double> {
  static inline double lower_bound() {
    return at::numeric_limits<double>::lower_bound();
  }
  static inline double upper_bound() {
    return at::numeric_limits<double>::upper_bound();
  }

  static inline bool lt(double a, double b) {
    return DPCPP::isless(a, b);
  }
  static inline bool le(double a, double b) {
    return DPCPP::islessequal(a, b);
  }
  static inline bool gt(double a, double b) {
    return DPCPP::isgreater(a, b);
  }
  static inline bool ge(double a, double b) {
    return DPCPP::isgreaterequal(a, b);
  }
  static inline bool eq(double a, double b) {
    return DPCPP::isequal(a, b);
  }
  static inline bool ne(double a, double b) {
    return DPCPP::isnotequal(a, b);
  }

  static inline double exp(double a) {
    return DPCPP::exp(a);
  }
  static inline double exp10(double a) {
    return DPCPP::exp10(a);
  }
  static inline double log(double a) {
    return DPCPP::log(a);
  }
  static inline double log10(double a) {
    return DPCPP::log10(a);
  }
  static inline double log1p(double a) {
    return DPCPP::log1p(a);
  }
  static inline double log2(double a) {
    return DPCPP::log2(a);
  }
  static inline double expm1(double a) {
    return DPCPP::expm1(a);
  }

  static inline double neg(double a) {
    return -a;
  }
  static inline double sin(double a) {
    return DPCPP::sin(a);
  }
  static inline double cos(double a) {
    return DPCPP::cos(a);
  }
  static inline double sqrt(double a) {
    return DPCPP::sqrt(a);
  }
  static inline double rsqrt(double a) {
    return DPCPP::rsqrt(a);
  }
  static inline double ceil(double a) {
    return DPCPP::ceil(a);
  }
  static inline double floor(double a) {
    return DPCPP::floor(a);
  }
  static inline double trunc(double a) {
    return DPCPP::trunc(a);
  }
  static inline double acos(double a) {
    return DPCPP::acos(a);
  }
  static inline double cosh(double a) {
    return DPCPP::cosh(a);
  }
  static inline double asin(double a) {
    return DPCPP::asin(a);
  }
  static inline double sinh(double a) {
    return DPCPP::sinh(a);
  }
  static inline double tan(double a) {
    return DPCPP::tan(a);
  }
  static inline double atan(double a) {
    return DPCPP::atan(a);
  }
  static inline double tanh(double a) {
    return DPCPP::tanh(a);
  }
  static inline double erf(double a) {
    return DPCPP::erf(a);
  }
  static inline double erfc(double a) {
    return DPCPP::erfc(a);
  }
  static inline double round(double a) {
    return DPCPP::round(a);
  }

  static inline double frac(double a) {
    return a - DPCPP::trunc(a);
  }
  static inline double cinv(double a) {
    return 1.0f / a;
  }
  static inline double add(double a, double b) {
    return a + b;
  }
  static inline double div(double a, double b) {
    return a / b;
  }
  static inline double mul(double a, double b) {
    return a * b;
  }
  static inline double sub(double a, double b) {
    return a - b;
  }
  static inline double pow(double a, double b) {
    return DPCPP::pow(a, b);
  }
  static inline double atan2(double a, double b) {
    return DPCPP::atan2(a, b);
  }
  static inline double min(double a, double b) {
    return DPCPP::fmin(a, b);
  }
  static inline double max(double a, double b) {
    return DPCPP::fmax(a, b);
  }
  static inline double abs(double a) {
    return DPCPP::fabs(a);
  }
  static inline double fabs(double a) {
    return DPCPP::fabs(a);
  }
  static inline bool isnan(double a) {
    return DPCPP::isnan(a);
  }
  static inline double sgn(double a) {
    return sgni<double>(a);
  }
};

template <typename In, typename Out>
struct ScalarConvert {
  static Out to(const In v) {
    return (Out)v;
  }
};

template <typename T, typename U>
T scalar_cast(U u) {
  return ScalarConvert<U, T>::to(u);
}

template <typename T>
DPCPP_BOTH inline T Min(T a, T b) {
  return (a < b) ? a : b;
}

template <typename T>
DPCPP_BOTH inline T Max(T a, T b) {
  return (a > b) ? a : b;
}

template <typename T>
DPCPP_BOTH inline T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

/**
 *    Computes ceil(a / b) * b; i.e., rounds up `a` to the next highest
 *       multiple of b
 *       */

template <typename T>
DPCPP_BOTH inline T RoundUp(T a, T b) {
  return CeilDiv(a, b) * b;
}
