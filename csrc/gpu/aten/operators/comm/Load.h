#pragma once

#include <c10/macros/Macros.h>
#include <cstring>

namespace at {
namespace AtenIpexTypeXPU {

template <typename T>
struct LoadImpl {
  DPCPP_BOTH static T apply(const void* src) {
    return *reinterpret_cast<const T*>(src);
  }
};

template <>
struct LoadImpl<bool> {
  DPCPP_BOTH static bool apply(const void* src) {
    static_assert(sizeof(bool) == sizeof(char), "");
    // NOTE: [Loading boolean values]
    // Protect against invalid boolean values by loading as a byte
    // first, then converting to bool (see gh-54789).
    return *reinterpret_cast<const unsigned char*>(src);
  }
};

template <typename T>
DPCPP_BOTH T load(const void* src) {
  return LoadImpl<T>::apply(src);
}

template <typename scalar_t>
DPCPP_BOTH scalar_t load(const scalar_t* src) {
  return LoadImpl<scalar_t>::apply(src);
}

} // namespace AtenIpexTypeXPU
} // namespace at
