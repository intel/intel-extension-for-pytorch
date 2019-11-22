#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorMathCompareT.h"
#else

THSYCL_API void THSYCLTensor_(ltTensor)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(gtTensor)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(leTensor)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(geTensor)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(eqTensor)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(neTensor)(THSYCLState *state, THSyclBoolTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);

THSYCL_API void THSYCLTensor_(ltTensorT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(gtTensorT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(leTensorT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(geTensorT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(eqTensorT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(neTensorT)(THSYCLState *state, THSYCLTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);

THSYCL_API void THSYCLTensor_(ltTensorByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(gtTensorByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(leTensorByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(geTensorByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(eqTensorByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);
THSYCL_API void THSYCLTensor_(neTensorByte)(THSYCLState *state, THSyclByteTensor *self_, THSYCLTensor *src1, THSYCLTensor *src2);

#endif

