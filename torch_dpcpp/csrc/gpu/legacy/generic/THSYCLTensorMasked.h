#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensorMasked.h"
#else

THSYCL_API void THSYCLTensor_(maskedFill)(THSYCLState *state,
                                          THSYCLTensor *tensor,
                                          THSyclByteTensor *mask,
                                          scalar_t value);

THSYCL_API void THSYCLTensor_(maskedSelect)(THSYCLState* state,
                                            THSYCLTensor* tensor,
                                            THSYCLTensor* src,
                                            THSyclByteTensor* mask);

THSYCL_API void THSYCLTensor_(maskedFillBool)(THSYCLState *state,
                                    THSYCLTensor *tensor,
                                    THSyclBoolTensor *mask,
                                    scalar_t value);

THSYCL_API void THSYCLTensor_(maskedCopy)(THSYCLState* state,
                                    THSYCLTensor *tensor,
                                    THSyclByteTensor *mask,
                                    THSYCLTensor *src);

THSYCL_API void THSYCLTensor_(maskedCopyBool)(THSYCLState* state,
                                    THSYCLTensor *tensor,
                                    THSyclBoolTensor *mask,
                                    THSYCLTensor *src);

THSYCL_API void THSYCLTensor_(maskedSelect)(THSYCLState* state,
                                    THSYCLTensor* tensor,
                                    THSYCLTensor* src,
                                    THSyclByteTensor* mask);

THSYCL_API void THSYCLTensor_(maskedSelectBool)(THSYCLState* state,
                                    THSYCLTensor* tensor,
                                    THSYCLTensor* src,
                                    THSyclBoolTensor* mask);
#endif
