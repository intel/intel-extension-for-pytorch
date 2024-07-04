/*******************************************************************************
 * Copyright 2016-2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/*
Centralized header file for preprocessor macros and constants
used throughout the codebase.
*/

#pragma once

#include <sycl/sycl.hpp>

#define DS_HD_INLINE __inline__ __attribute__((always_inline))
#define DS_D_INLINE __inline__ __attribute__((always_inline))

// constexpr variant of warpSize for templating
constexpr int hw_warp_size = 32;

#define HALF_PRECISION_AVAILABLE = 1

inline int next_pow2(const int val) {
  int rounded_val = val - 1;
  rounded_val |= rounded_val >> 1;
  rounded_val |= rounded_val >> 2;
  rounded_val |= rounded_val >> 4;
  rounded_val |= rounded_val >> 8;
  return rounded_val + 1;
}

namespace ds {
// pow functions overload.
inline float pow(const float a, const int b) {
  return sycl::pown(a, b);
}
inline double pow(const double a, const int b) {
  return sycl::pown(a, b);
}
inline float pow(const float a, const float b) {
  return sycl::pow(a, b);
}
inline double pow(const double a, const double b) {
  return sycl::pow(a, b);
}
template <typename T, typename U>
inline typename std::enable_if_t<std::is_floating_point_v<T>, T> pow(
    const T a,
    const U b) {
  return sycl::pow(a, static_cast<T>(b));
}
template <typename T, typename U>
inline typename std::enable_if_t<!std::is_floating_point_v<T>, double> pow(
    const T a,
    const U b) {
  return sycl::pow(static_cast<double>(a), static_cast<double>(b));
}

inline void has_capability_or_fail(
    const sycl::device& dev,
    const std::initializer_list<sycl::aspect>& props) {
  for (const auto& it : props) {
    if (dev.has(it))
      continue;
    switch (it) {
      case sycl::aspect::fp64:
        throw std::runtime_error(
            "'double' is not supported in '" +
            dev.get_info<sycl::info::device::name>() + "' device");
        break;
      case sycl::aspect::fp16:
        throw std::runtime_error(
            "'half' is not supported in '" +
            dev.get_info<sycl::info::device::name>() + "' device");
        break;
      default:
#define __SYCL_ASPECT(ASPECT, ID) \
  case sycl::aspect::ASPECT:      \
    return #ASPECT;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE) __SYCL_ASPECT(ASPECT, ID)
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
        auto getAspectNameStr = [](sycl::aspect AspectNum) -> std::string {
          switch (AspectNum) {
#include <sycl/info/aspects.def>
#include <sycl/info/aspects_deprecated.def>
            default:
              return "unknown aspect";
          }
        };
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
        throw std::runtime_error(
            "'" + getAspectNameStr(it) + "' is not supported in '" +
            dev.get_info<sycl::info::device::name>() + "' device");
    }
    break;
  }
}
} // namespace ds
