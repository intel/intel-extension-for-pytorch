#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorCopy.h"
#else

THSYCL_API void THSYCLTensor_(copy)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(copyIgnoringOverlaps)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);

THSYCL_API void THSYCLTensor_(copyAsyncCPU)(THSYCLState *state, THSYCLTensor *self, THTensor *src);
THSYCL_API void THTensor_(copyAsyncCuda)(THSYCLState *state, THTensor *self, THSYCLTensor *src);

#endif

