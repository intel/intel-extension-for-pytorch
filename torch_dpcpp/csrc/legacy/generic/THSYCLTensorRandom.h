#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorRandom.h"
#else

#if defined(THSYCL_REAL_IS_FLOAT) || defined(THSYCL_REAL_IS_DOUBLE) || defined(THSYCL_REAL_IS_HALF)
THSYCL_API void THSYCLTensor_(uniform)(THSYCLState* state, THSYCLTensor *self, at::Generator *_generator, double a, double b);
THSYCL_API void THSYCLTensor_(normal)(THSYCLState* state, THSYCLTensor *self, at::Generator *_generator, double mean, double stdv);
#endif

#endif
