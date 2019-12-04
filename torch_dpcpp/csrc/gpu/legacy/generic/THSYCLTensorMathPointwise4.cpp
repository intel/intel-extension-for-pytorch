#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorMathPointwise4.cpp"
#else

#include "THSYCLTensorMathPointwise.hpp"

#if !defined(THSYCL_REAL_IS_BOOL)

IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(neg, THSYCLNumerics<scalar_t>::neg, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(abs, THSYCLNumerics<scalar_t>::abs, Real)

#if defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_DOUBLE) || defined(THSYCL_REAL_IS_HALF)

IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(ceil,  THSYCLNumerics<scalar_t>::ceil, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(floor, THSYCLNumerics<scalar_t>::floor, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(trunc, THSYCLNumerics<scalar_t>::trunc, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(round, THSYCLNumerics<scalar_t>::round, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(frac,  THSYCLNumerics<scalar_t>::frac,  Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(cinv,  THSYCLNumerics<scalar_t>::cinv,  Real)

IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(exp,   THSYCLNumerics<scalar_t>::exp, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(expm1, THSYCLNumerics<scalar_t>::expm1, Real)

IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(erf,   THSYCLNumerics<scalar_t>::erf,   Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(erfc,  THSYCLNumerics<scalar_t>::erfc,  Real)

#endif

#endif

#endif
