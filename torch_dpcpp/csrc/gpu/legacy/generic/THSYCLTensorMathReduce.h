#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorMathReduce.h"
#else

THSYCL_API scalar_t THSYCLTensor_(maxall)(THSYCLState *state, THSYCLTensor *self);

THSYCL_API void THSYCLTensor_(max)(THSYCLState *state,
                             THSYCLTensor *values,
                             THSyclLongTensor *indices,
                             THSYCLTensor *src, int dim, int keepdim);


THSYCL_API scalar_t THSYCLTensor_(minall)(THSYCLState *state, THSYCLTensor *self);

THSYCL_API void THSYCLTensor_(min)(THSYCLState *state,
                             THSYCLTensor *values,
                             THSyclLongTensor *indices,
                             THSYCLTensor *src, int dim, int keepdim);

THSYCL_API accreal THSYCLTensor_(dist)(THSYCLState* state, THSYCLTensor* self,
                             THSYCLTensor* src, scalar_t _value);

THSYCL_API accreal THSYCLTensor_(sumall)(THSYCLState *state, THSYCLTensor *self);

THSYCL_API scalar_t THSYCLTensor_(medianall)(THSYCLState *state, THSYCLTensor *self);

THSYCL_API void THSYCLTensor_(median)(THSYCLState *state,
                                THSYCLTensor *values,
                                THSyclLongTensor *indices,
                                THSYCLTensor *src, int dim, int keepdim);


#endif
