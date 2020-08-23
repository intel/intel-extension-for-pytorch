#ifndef ALGORITHM_INC
#define ALGORITHM_INC

#include <core/DPCPP.h>
#include <utils/General.h>
#include <utils/Numerics.h>

// alternative for thrust::tabulate
template <typename ForwardIterator, typename UnaryOp>
static inline void dpcpp_tabulate(
    ForwardIterator begin,
    ForwardIterator end,
    UnaryOp unary_op) {
  for (int i = 0; (begin + i) != end; ++i) {
    *(begin + i) = unary_op(i);
  }
}

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

// alternative for thrust::exclusive_scan
template <
    typename InputIterator,
    typename OutputIterator,
    typename T,
    typename AssociativeOperator>
static inline void dpcpp_exclusive_scan(
    InputIterator first,
    InputIterator last,
    OutputIterator result,
    T init,
    AssociativeOperator binary_op) {
  for (size_t i = 0; (first + i) != last; ++i) {
    if (i == 0) {
      *result = init;
    } else {
      *(result + i) = binary_op(*(result + i - 1), *(first + i - 1));
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

#endif
