#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorMathPairwise.h"
#else

THSYCL_API int THSYCLTensor_(equal)(THSYCLState* state, THSYCLTensor* self_, THSYCLTensor* src_);
THSYCL_API void THSYCLTensor_(bitand)(THSYCLState* state, THSYCLTensor *self_, THSYCLTensor *src_, scalar_t value);
THSYCL_API void THSYCLTensor_(bitor)(THSYCLState* state, THSYCLTensor *self_, THSYCLTensor *src_, scalar_t value);
THSYCL_API void THSYCLTensor_(bitxor)(THSYCLState* state, THSYCLTensor* self_, THSYCLTensor* src_, scalar_t value);

#if !defined(THSYCL_REAL_IS_BOOL)
THSYCL_API void THSYCLTensor_(lshift)(THSYCLState* state, THSYCLTensor* self_, THSYCLTensor* src_, scalar_t value);
THSYCL_API void THSYCLTensor_(rshift)(THSYCLState* state, THSYCLTensor* self_, THSYCLTensor* src_, scalar_t value);
THSYCL_API void THSYCLTensor_(fmod)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(remainder)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, scalar_t value);
#endif

#endif
