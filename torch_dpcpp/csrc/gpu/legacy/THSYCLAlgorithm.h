#ifndef TH_SYCL_ALGORITHM_INC
#define TH_SYCL_ALGORITHM_INC

#include <core/SYCL.h>
#include <legacy/THSYCLGeneral.h>
#include <legacy/THSYCLNumerics.h>

// alternative for cuda thrust::tabulate
template <typename ForwardIterator, typename UnaryOp>
void sycl_tabulate(ForwardIterator begin, ForwardIterator end, UnaryOp unary_op) {
  for (int i = 0; (begin + i) != end; ++i) {
    *(begin + i) = unary_op(i);
  }
}

// alternative for cuda thrust::inclusive_scan
template <typename InputIterator, typename OutputIterator, typename AssociativeOperator>
void sycl_inclusive_scan(InputIterator first, InputIterator last, OutputIterator result, AssociativeOperator binary_op) {
  for (int i = 0; (first + i) != last; ++i) {
    if (i == 0) *result = *first;
    else {
      *(result + i) = binary_op(*(result + i - 1), *(first + i));
    }
  }
}

// alternative for cuda thrust::exclusive_scan
template <typename InputIterator, typename OutputIterator, typename T, typename AssociativeOperator>
void sycl_exclusive_scan(InputIterator first,
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
DP_DEVICE struct sycl_identity {
  T const &operator()(T const &t) const {
    return t;
  }

  T &operator()(T &t) const {
    return t;
  }
};

#endif
