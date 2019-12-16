#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLStorage.h"
#else
#define THSYCLStorage THStorage

// These used to be distinct types; for some measure of backwards compatibility and documentation
// alias these to the single THCStorage type.
#define THSyclStorage       THSYCLStorage
#define THSyclDoubleStorage THSYCLStorage
#define THSyclHalfStorage   THSYCLStorage
#define THSyclByteStorage   THSYCLStorage
#define THSyclCharStorage   THSYCLStorage
#define THSyclShortStorage  THSYCLStorage
#define THSyclIntStorage    THSYCLStorage
#define THSyclLongStorage   THSYCLStorage

THSYCL_API scalar_t* THSYCLStorage_(data)(THSYCLState *state, const THSYCLStorage*);
THSYCL_API ptrdiff_t THSYCLStorage_(size)(THSYCLState *state, const THSYCLStorage*);
THSYCL_API int THSYCLStorage_(elementSize)(THSYCLState *state);

/* slow access --checks everything */
THSYCL_API  void THSYCLStorage_(set)(THSYCLState *state, THSYCLStorage*, ptrdiff_t, scalar_t);
THSYCL_API scalar_t THSYCLStorage_(get)(THSYCLState *state, const THSYCLStorage*, ptrdiff_t);

THSYCL_API THSYCLStorage* THSYCLStorage_(new)(THSYCLState *state);
THSYCL_API THSYCLStorage* THSYCLStorage_(newWithSize)(THSYCLState *state, ptrdiff_t size);
THSYCL_API THSYCLStorage* THSYCLStorage_(newWithSize1)(THSYCLState *state, scalar_t);
THSYCL_API THSYCLStorage* THSYCLStorage_(newWithSize2)(THSYCLState *state, scalar_t, scalar_t);
THSYCL_API THSYCLStorage* THSYCLStorage_(newWithSize3)(THSYCLState *state, scalar_t, scalar_t, scalar_t);
THSYCL_API THSYCLStorage* THSYCLStorage_(newWithSize4)(THSYCLState *state, scalar_t, scalar_t, scalar_t, scalar_t);
THSYCL_API THSYCLStorage* THSYCLStorage_(newWithMapping)(THSYCLState *state, const char* filename, ptrdiff_t size, int shared);

#ifdef __cplusplus
THSYCL_API THSYCLStorage* THSYCLStorage_(newWithAllocator)(
    THSYCLState *state, ptrdiff_t size,
    c10::Allocator* allocator);
THSYCL_API THSYCLStorage* THSYCLStorage_(newWithDataAndAllocator)(
    THSYCLState *state, c10::DataPtr&& data, ptrdiff_t size,
    c10::Allocator* allocator);
#endif

THSYCL_API void THSYCLStorage_(setFlag)(THSYCLState *state, THSYCLStorage *storage, const char flag);
THSYCL_API void THSYCLStorage_(clearFlag)(THSYCLState *state, THSYCLStorage *storage, const char flag);
THSYCL_API void THSYCLStorage_(retain)(THSYCLState* state, THSYCLStorage *storage);

THSYCL_API void THSYCLStorage_(free)(THSYCLState* state, THSYCLStorage *storage);
THSYCL_API void THSYCLStorage_(resize)(THSYCLState* state, THSYCLStorage *storage,  ptrdiff_t size);
THSYCL_API void THSYCLStorage_(fill)(THSYCLState* state, THSYCLStorage *storage, scalar_t value);

THSYCL_API int THSYCLStorage_(getDevice)(THSYCLState* state, const THSYCLStorage* storage);


#endif
