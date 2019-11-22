#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorTopK.h"
#else

/* Returns the set of all kth smallest (or largest) elements, depending */
/* on `dir` */
THSYCL_API void THSYCLTensor_(topk)(THSYCLState* state,
                               THSYCLTensor* topK,
                               THSyclLongTensor* indices,
                               THSYCLTensor* input,
                               int64_t k, int dim, int dir, int sorted);

#endif
