#pragma once

#include <ATen/ATen.h>

#include "General.h"

template <typename T>
struct KeyTraits {};

template <>
struct KeyTraits<NullType> {
  using Type = uint32_t;
  static inline Type convert(float v) {
    return 0;
  }
  static inline NullType deconvert(Type v) {
    return NullType();
  }
  static inline unsigned int endbit() {
    return 0;
  }
};

template <>
struct KeyTraits<float> {
  using Type = uint32_t;
  static inline Type convert(float v) {
    Type x = *((Type*)&v);
    Type mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
    return (x ^ mask);
  }
  static inline float deconvert(Type v) {
    Type mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
    auto v_de = v ^ mask;
    return *((float*)&v_de);
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<bool> {
  using Type = bool;
  static inline Type convert(bool v) {
    return v;
  }
  static inline bool deconvert(Type v) {
    return v;
  }
  static inline int endbit() {
    return 1;
  }
};

template <>
struct KeyTraits<uint8_t> {
  using Type = uint8_t;
  static inline Type convert(uint8_t v) {
    return v;
  }
  static inline uint8_t deconvert(Type v) {
    return v;
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<int8_t> {
  using Type = uint8_t;
  static inline Type convert(int8_t v) {
    return 128u + v;
  }
  static inline int8_t deconvert(Type v) {
    return v - 128;
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<int16_t> {
  using Type = uint16_t;
  static inline Type convert(int16_t v) {
    return 32768u + v;
  }
  static inline int16_t deconvert(Type v) {
    return v - 32768;
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<int32_t> {
  using Type = uint32_t;
  static inline Type convert(int32_t v) {
    return 2147483648u + v;
  }
  static inline int32_t deconvert(Type v) {
    return v - 2147483648u;
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<int64_t> {
  using Type = uint64_t;
  static inline Type convert(int64_t v) {
    return 9223372036854775808ull + v;
  }
  static inline int64_t deconvert(Type v) {
    return v - 9223372036854775808ull;
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<double> {
  using Type = uint64_t;
  static inline Type convert(double v) {
    Type x = *((Type*)&v);
    Type mask = -((x >> 63)) | 0x8000000000000000;
    return (x ^ mask);
  }
  static inline double deconvert(Type v) {
    Type mask = ((v >> 63) - 1) | 0x8000000000000000;
    auto v_de = v ^ mask;
    return *((double*)&v_de);
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<at::Half> {
  using Type = uint16_t;
  static inline Type convert(at::Half v) {
    Type x = *((Type*)&v);
    Type mask = -((x >> 15)) | 0x8000;
    return (x ^ mask);
  }
  static inline at::Half deconvert(Type v) {
    Type mask = ((v >> 15) - 1) | 0x8000;
    auto v_de = v ^ mask;
    return *((at::Half*)&v_de);
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<at::BFloat16> {
  using Type = uint16_t;
  static inline Type convert(at::BFloat16 v) {
    Type x = *((Type*)&v);
    Type mask = -((x >> 15)) | 0x8000;
    return (x ^ mask);
  }
  static inline at::BFloat16 deconvert(Type v) {
    Type mask = ((v >> 15) - 1) | 0x8000;
    auto v_de = v ^ mask;
    return *((at::BFloat16*)&v_de);
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};
