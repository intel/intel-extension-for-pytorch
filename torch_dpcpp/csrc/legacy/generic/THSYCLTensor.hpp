#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensor.hpp"
#else

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

// NOTE: functions exist here only to support dispatch via Declarations.cwrap.  You probably don't want to put
// new functions in here, they should probably be un-genericized.

THSYCL_API void THSYCLTensor_(setStorage)(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage_, ptrdiff_t storageOffset_,
                                    at::IntArrayRef size_, at::IntArrayRef stride_);
THSYCL_API THSYCLTensor *THSYCLTensor_(newView)(THSYCLState *state, THSYCLTensor *tensor, at::IntArrayRef size);
/* strides.data() might be nullptr */
THSYCL_API THSYCLTensor *THSYCLTensor_(newWithStorage)(THSYCLState *state, THSYCLStorage *storage, ptrdiff_t storageOffset,
                                              at::IntArrayRef sizes, at::IntArrayRef strides);

THSYCL_API void THSYCLTensor_(resize)(THSYCLState *state, THSYCLTensor *self, at::IntArrayRef size, at::IntArrayRef stride);
THSYCL_API THSYCLTensor *THSYCLTensor_(newWithSize)(THSYCLState *state, at::IntArrayRef size, at::IntArrayRef stride);

#endif

