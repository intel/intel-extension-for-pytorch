#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorMath.h"
#else

THSYCL_API void THSYCLTensor_(fill)(THSYCLState *state, THSYCLTensor *self, scalar_t value);
THSYCL_API void THSYCLTensor_(zero)(THSYCLState *state, THSYCLTensor *self);
THSYCL_API void THSYCLTensor_(nonzero)(THSYCLState* state, THSyclLongTensor *tensor, THSYCLTensor *self);
THSYCL_API void THSYCLTensor_(diag)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src_, int64_t k);
THSYCL_API void THSYCLTensor_(tril)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int64_t k);
THSYCL_API void THSYCLTensor_(triu)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int64_t k);

THSYCL_API void THSYCLTensor_(cat)(THSYCLState *state, THSYCLTensor *r_, THSYCLTensor *ta, THSYCLTensor *tb, int dimension);

THSYCL_API void THSYCLTensor_(catArray)(THSYCLState *state, THSYCLTensor *result, THSYCLTensor **inputs, int numInputs, int dimension);

THSYCL_API accreal THSYCLTensor_(trace)(THSYCLState* state, THSYCLTensor* src_);
#endif
