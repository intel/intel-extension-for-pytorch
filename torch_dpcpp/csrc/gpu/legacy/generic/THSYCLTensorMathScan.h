#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorMathScan.h"
#else

THSYCL_API void THSYCLTensor_(cumprod)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension);
THSYCL_API void THSYCLTensor_(cumsum)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension);


#endif
