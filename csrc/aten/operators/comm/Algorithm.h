#pragma once

#include <utils/DPCPP.h>
#include <tuple>
#include "Numerics.h"

// alternative for thrust::inclusive_scan
template <
    typename InputIterator,
    typename OutputIterator,
    typename AssociativeOperator>
static inline void dpcpp_inclusive_scan(
    InputIterator first,
    InputIterator last,
    OutputIterator result,
    AssociativeOperator binary_op) {
  for (int i = 0; (first + i) != last; ++i) {
    if (i == 0)
      *result = *first;
    else {
      *(result + i) = binary_op(*(result + i - 1), *(first + i));
    }
  }
}

// alternative for cude thrust::identity
template <typename T>
DPCPP_DEVICE struct dpcpp_identity {
  T const& operator()(T const& t) const {
    return t;
  }

  T& operator()(T& t) const {
    return t;
  }
};

template <typename T>
DPCPP_DEVICE struct dpcpp_transformation {
  std::tuple<T, decltype(std::ignore)> operator()(T& x) const {
    return std::forward_as_tuple(x, std::ignore);
  }
};
