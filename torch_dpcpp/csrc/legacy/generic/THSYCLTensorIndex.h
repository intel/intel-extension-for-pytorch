#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorIndex.h"
#else

THSYCL_API void THSYCLTensor_(indexSelect)(THSYCLState *state, THSYCLTensor *dst, THSYCLTensor *src, int dim, THSYCLTensor *indices);

THSYCL_API void THSYCLTensor_(indexCopy)(THSYCLState* state, THSYCLTensor* dst, int dim, THSyclLongTensor* indices, THSYCLTensor* src);

THSYCL_API void THSYCLTensor_(take)(THSYCLState* state, THSYCLTensor* dst, THSYCLTensor* src, THSyclLongTensor* index);

THSYCL_API void THSYCLTensor_(indexAdd)(THSYCLState *state, THSYCLTensor *dst, int dim, THSyclLongTensor *indices, THSYCLTensor *src);

THSYCL_API void THSYCLTensor_(indexFill)(THSYCLState* state, THSYCLTensor* dst, int dim, THSyclLongTensor* indices, scalar_t val);
#endif
