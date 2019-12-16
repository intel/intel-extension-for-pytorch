#pragma once

// A fixed-size array type usable from SYCL kernels.

#include <c10/macros/Macros.h>

namespace at { namespace sycl {

template <typename T, int size>
struct alignas(16) Array {
  T data[size];

  T operator[](int i) const {
    return data[i];
  }
  T& operator[](int i) {
    return data[i];
  }

  Array() = default;
  Array(const Array&) = default;
  Array& operator=(const Array&) = default;

  // Fill the array with x.
  Array(T x) {
    for (int i = 0; i < size; i++) {
      data[i] = x;
    }
  }
};

}}
