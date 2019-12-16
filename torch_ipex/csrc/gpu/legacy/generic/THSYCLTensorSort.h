#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorSort.h"
#else

/* Performs an in-place sort of (keys, values). Only works for slice sizes
   <= 2048 at the moment (slice size == size of keys/values dim `dim`) */
#ifdef __cplusplus
THSYCL_API void THSYCLTensor_(sortKeyValueInplace)(THSYCLState* state,
                                             THSYCLTensor* keys,
                                             THSyclLongTensor* values,
                                             int dim, bool dir);
#else
THSYCL_API void THSYCLTensor_(sortKeyValueInplace)(THSYCLState* state,
                                             THSYCLTensor* keys,
                                             THSyclLongTensor* values,
                                             int dim, int order);
#endif

/* Performs an out-of-place sort of `input`, returning the per-slice indices
   in `indices` and the sorted values in `sorted` */
THSYCL_API void THSYCLTensor_(sort)(THSYCLState* state,
                              THSYCLTensor* sorted,
                              THSyclLongTensor* indices,
                              THSYCLTensor* input,
                              int dim, int order);

#endif

