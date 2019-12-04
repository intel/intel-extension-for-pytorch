#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorMathPointwise5.cpp"
#else

#include "THSYCLTensorMathPointwise.hpp"

#if !defined(THSYCL_REAL_IS_BOOL)

#if defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_DOUBLE) || defined(THSYCL_REAL_IS_HALF)

IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(sin,   THSYCLNumerics<scalar_t>::sin, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(asin,  THSYCLNumerics<scalar_t>::asin, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(sinh,  THSYCLNumerics<scalar_t>::sinh, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(cos,   THSYCLNumerics<scalar_t>::cos, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(acos,  THSYCLNumerics<scalar_t>::acos, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(cosh,  THSYCLNumerics<scalar_t>::cosh, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(tan,   THSYCLNumerics<scalar_t>::tan, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(atan,  THSYCLNumerics<scalar_t>::atan, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(tanh,  THSYCLNumerics<scalar_t>::tanh,  Real)

IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(log,   THSYCLNumerics<scalar_t>::log, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(log10, THSYCLNumerics<scalar_t>::log10, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(log1p, THSYCLNumerics<scalar_t>::log1p, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(log2,  THSYCLNumerics<scalar_t>::log2, Real)

IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(sqrt,  THSYCLNumerics<scalar_t>::sqrt, Real)
IMPLEMENT_SYCL_TENSOR_BASIC_FUNC(rsqrt, THSYCLNumerics<scalar_t>::rsqrt, Real)

#endif

#endif

#endif
