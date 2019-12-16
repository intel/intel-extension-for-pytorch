#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorMode.h"
#else

/* Returns the mode, and index of the mode, for the set of values
 * along a given dimension in the input tensor. */
THSYCL_API void THSYCLTensor_(mode)(THSYCLState* state,
                                    THSYCLTensor* values,
                                    THSyclLongTensor* indices,
                                    THSYCLTensor* input,
                                    int dimension,
                                    int keepdim);

#endif // THSYCL_GENERIC_FILE
