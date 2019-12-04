#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorMathBlas.h"
#else

THSYCL_API void THSYCLTensor_(addmv)(THSYCLState* state, THSYCLTensor* r_, scalar_t beta, THSYCLTensor* t, scalar_t alpha, THSYCLTensor* mat, THSYCLTensor* vec);
THSYCL_API void THSYCLTensor_(addr)(THSYCLState* state, THSYCLTensor* r_, scalar_t beta, THSYCLTensor* t, scalar_t alpha, THSYCLTensor* vec1, THSYCLTensor* vec2);
THSYCL_API void THSYCLTensor_(addmm)(THSYCLState *state, THSYCLTensor *self, scalar_t beta, THSYCLTensor *t, scalar_t alpha, THSYCLTensor *mat1, THSYCLTensor *mat2);
THSYCL_API void THSYCLTensor_(addbmm)(THSYCLState* state, THSYCLTensor* result, scalar_t beta, THSYCLTensor* t, scalar_t alpha, THSYCLTensor* batch1, THSYCLTensor* batch2);

THSYCL_API void THSYCLTensor_(baddbmm)(THSYCLState *state, THSYCLTensor *result, scalar_t beta, THSYCLTensor *t, scalar_t alpha, THSYCLTensor *batch1, THSYCLTensor *batch2);

#endif

