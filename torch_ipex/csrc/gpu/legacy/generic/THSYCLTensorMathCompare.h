#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorMathCompare.h"
#else

THSYCL_API void THSYCLTensor_(ltValue)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(gtValue)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(leValue)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(geValue)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(eqValue)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(neValue)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src, scalar_t value);

THSYCL_API void THSYCLTensor_(ltValueT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(gtValueT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(leValueT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(geValueT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(eqValueT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(neValueT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src, scalar_t value);


THSYCL_API void THSYCLTensor_(ltValueByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(gtValueByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(leValueByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(geValueByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(eqValueByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src, scalar_t value);
THSYCL_API void THSYCLTensor_(neValueByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src, scalar_t value);

#endif
