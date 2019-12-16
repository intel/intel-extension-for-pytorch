#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensor.h"
#else

#define THSYCLTensor THTensor

// These used to be distinct types; for some measure of backwards compatibility and documentation
// alias these to the single THSYCLTensor type.
#define THSyclTensor THSYCLTensor
#define THSyclDoubleTensor THSYCLTensor
#define THSyclHalfTensor THSYCLTensor
#define THSyclByteTensor THSYCLTensor
#define THSyclCharTensor THSYCLTensor
#define THSyclShortTensor THSYCLTensor
#define THSyclIntTensor THSYCLTensor
#define THSyclLongTensor THSYCLTensor
#define THSyclBoolTensor THSYCLTensor
#define THSyclBFloat16Tensor THSYCLTensor

/**** access methods ****/
THSYCL_API THSYCLStorage* THSYCLTensor_(storage)(THSYCLState *state, const THSYCLTensor *self);
THSYCL_API ptrdiff_t THSYCLTensor_(storageOffset)(THSYCLState *state, const THSYCLTensor *self);

// See [NOTE: nDimension vs nDimensionLegacyNoScalars vs nDimensionLegacyAll]
THSYCL_API int THSYCLTensor_(nDimension)(THSYCLState *state, const THSYCLTensor *self);
THSYCL_API int THSYCLTensor_(nDimensionLegacyNoScalars)(THSYCLState *state, const THSYCLTensor *self);
THSYCL_API int THSYCLTensor_(nDimensionLegacyAll)(THSYCLState *state, const THSYCLTensor *self);

THSYCL_API int64_t THSYCLTensor_(size)(THSYCLState *state, const THSYCLTensor *self, int dim);
THSYCL_API int64_t THSYCLTensor_(sizeLegacyNoScalars)(THSYCLState *state, const THSYCLTensor *self, int dim);
THSYCL_API int64_t THSYCLTensor_(stride)(THSYCLState *state, const THSYCLTensor *self, int dim);
THSYCL_API int64_t THSYCLTensor_(strideLegacyNoScalars)(THSYCLState *state, const THSYCLTensor *self, int dim);
THSYCL_API scalar_t *THSYCLTensor_(data)(THSYCLState *state, const THSYCLTensor *self);

THSYCL_API void THSYCLTensor_(setFlag)(THSYCLState *state, THSYCLTensor *self, const char flag);
THSYCL_API void THSYCLTensor_(clearFlag)(THSYCLState *state, THSYCLTensor *self, const char flag);


/**** creation methods ****/
THSYCL_API THSYCLTensor *THSYCLTensor_(new)(THSYCLState *state);
THSYCL_API THSYCLTensor *THSYCLTensor_(newWithTensor)(THSYCLState *state, THSYCLTensor *tensor);
THSYCL_API THSYCLTensor *THSYCLTensor_(newWithStorage1d)(THSYCLState *state, THSYCLStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_);
THSYCL_API THSYCLTensor *THSYCLTensor_(newWithStorage2d)(THSYCLState *state, THSYCLStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_,
                                int64_t size1_, int64_t stride1_);
THSYCL_API THSYCLTensor *THSYCLTensor_(newWithStorage3d)(THSYCLState *state, THSYCLStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_,
                                int64_t size1_, int64_t stride1_,
                                int64_t size2_, int64_t stride2_);
THSYCL_API THSYCLTensor *THSYCLTensor_(newWithStorage4d)(THSYCLState *state, THSYCLStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_,
                                int64_t size1_, int64_t stride1_,
                                int64_t size2_, int64_t stride2_,
                                int64_t size3_, int64_t stride3_);

/* stride might be NULL */
THSYCL_API THSYCLTensor *THSYCLTensor_(newWithSize1d)(THSYCLState *state, int64_t size0_);
THSYCL_API THSYCLTensor *THSYCLTensor_(newWithSize2d)(THSYCLState *state, int64_t size0_, int64_t size1_);
THSYCL_API THSYCLTensor *THSYCLTensor_(newWithSize3d)(THSYCLState *state, int64_t size0_, int64_t size1_, int64_t size2_);
THSYCL_API THSYCLTensor *THSYCLTensor_(newWithSize4d)(THSYCLState *state, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_);

THSYCL_API THSYCLTensor *THSYCLTensor_(newClone)(THSYCLState *state, THSYCLTensor *self);
THSYCL_API THSYCLTensor *THSYCLTensor_(newContiguous)(THSYCLState *state, THSYCLTensor *tensor);
THSYCL_API THSYCLTensor *THSYCLTensor_(newSelect)(THSYCLState *state, THSYCLTensor *tensor, int dimension_, int64_t sliceIndex_);
THSYCL_API THSYCLTensor *THSYCLTensor_(newNarrow)(THSYCLState *state, THSYCLTensor *tensor, int dimension_, int64_t firstIndex_, int64_t size_);
THSYCL_API THSYCLTensor *THSYCLTensor_(newTranspose)(THSYCLState *state, THSYCLTensor *tensor, int dimension1_, int dimension2_);

THSYCL_API THSYCLTensor *THSYCLTensor_(newUnfold)(THSYCLState *state, THSYCLTensor *tensor, int dimension_, int64_t size_, int64_t step_);
THSYCL_API THSYCLTensor *THSYCLTensor_(newFoldBatchDim)(THSYCLState *state, THSYCLTensor *input);

// resize* methods simply resize the storage. So they may not retain the current data at current indices.
// This is especially likely to happen when the tensor is not contiguous. In general, if you still need the
// values, unless you are doing some size and stride tricks, do not use resize*.
THSYCL_API void THSYCLTensor_(resizeNd)(THSYCLState *state, THSYCLTensor *tensor, int nDimension, const int64_t *size, const int64_t *stride);
THSYCL_API void THSYCLTensor_(resizeAs)(THSYCLState *state, THSYCLTensor *tensor, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(resize0d)(THSYCLState *state, THSYCLTensor *tensor);
THSYCL_API void THSYCLTensor_(resize1d)(THSYCLState *state, THSYCLTensor *tensor, int64_t size0_);
THSYCL_API void THSYCLTensor_(resize2d)(THSYCLState *state, THSYCLTensor *tensor, int64_t size0_, int64_t size1_);
THSYCL_API void THSYCLTensor_(resize3d)(THSYCLState *state, THSYCLTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_);
THSYCL_API void THSYCLTensor_(resize4d)(THSYCLState *state, THSYCLTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_);
THSYCL_API void THSYCLTensor_(resize5d)(THSYCLState *state, THSYCLTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_, int64_t size4_);

THSYCL_API void THSYCLTensor_(set)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(setStorageNd)(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage, ptrdiff_t storageOffset, int nDimension, const int64_t *size, const int64_t *stride);
THSYCL_API void THSYCLTensor_(setStorage1d)(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_);
THSYCL_API void THSYCLTensor_(setStorage2d)(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_,
                                    int64_t size1_, int64_t stride1_);
THSYCL_API void THSYCLTensor_(setStorage3d)(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_,
                                    int64_t size1_, int64_t stride1_,
                                    int64_t size2_, int64_t stride2_);
THSYCL_API void THSYCLTensor_(setStorage4d)(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_,
                                    int64_t size1_, int64_t stride1_,
                                    int64_t size2_, int64_t stride2_,
                                    int64_t size3_, int64_t stride3_);

THSYCL_API void THSYCLTensor_(narrow)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension_, int64_t firstIndex_, int64_t size_);
THSYCL_API void THSYCLTensor_(select)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension_, int64_t sliceIndex_);
THSYCL_API void THSYCLTensor_(transpose)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension1_, int dimension2_);
THSYCL_API void THSYCLTensor_(unfold)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension_, int64_t size_, int64_t step_);

THSYCL_API void THSYCLTensor_(squeeze)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_(squeeze1d)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension_);
THSYCL_API void THSYCLTensor_(unsqueeze1d)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension_);

THSYCL_API int THSYCLTensor_(isContiguous)(THSYCLState *state, const THSYCLTensor *self);
THSYCL_API int THSYCLTensor_(isSameSizeAs)(THSYCLState *state, const THSYCLTensor *self, const THSYCLTensor *src);
THSYCL_API int THSYCLTensor_(isSetTo)(THSYCLState *state, const THSYCLTensor *self, const THSYCLTensor *src);
THSYCL_API ptrdiff_t THSYCLTensor_(nElement)(THSYCLState *state, const THSYCLTensor *self);

THSYCL_API void THSYCLTensor_(retain)(THSYCLState *state, THSYCLTensor *self);
THSYCL_API void THSYCLTensor_(free)(THSYCLState *state, THSYCLTensor *self);
THSYCL_API void THSYCLTensor_(freeCopyTo)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *dst);

/* Slow access methods [check everything] */
THSYCL_API void THSYCLTensor_(set0d)(THSYCLState *state, THSYCLTensor *tensor, scalar_t value);
THSYCL_API void THSYCLTensor_(set1d)(THSYCLState *state, THSYCLTensor *tensor, int64_t x0, scalar_t value);
THSYCL_API void THSYCLTensor_(set2d)(THSYCLState *state, THSYCLTensor *tensor, int64_t x0, int64_t x1, scalar_t value);
THSYCL_API void THSYCLTensor_(set3d)(THSYCLState *state, THSYCLTensor *tensor, int64_t x0, int64_t x1, int64_t x2, scalar_t value);
THSYCL_API void THSYCLTensor_(set4d)(THSYCLState *state, THSYCLTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, scalar_t value);

THSYCL_API scalar_t THSYCLTensor_(get0d)(THSYCLState *state, const THSYCLTensor *tensor);
THSYCL_API scalar_t THSYCLTensor_(get1d)(THSYCLState *state, const THSYCLTensor *tensor, int64_t x0);
THSYCL_API scalar_t THSYCLTensor_(get2d)(THSYCLState *state, const THSYCLTensor *tensor, int64_t x0, int64_t x1);
THSYCL_API scalar_t THSYCLTensor_(get3d)(THSYCLState *state, const THSYCLTensor *tensor, int64_t x0, int64_t x1, int64_t x2);
THSYCL_API scalar_t THSYCLTensor_(get4d)(THSYCLState *state, const THSYCLTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3);

/* CUDA-specific functions */
THSYCL_API int THSYCLTensor_(getDevice)(THSYCLState *state, const THSYCLTensor *self);
THSYCL_API int THSYCLTensor_(checkGPU)(THSYCLState *state, unsigned int nTensors, ...);

/* debug methods */
THSYCL_API THSYCLDescBuff THSYCLTensor_(sizeDesc)(THSYCLState *state, const THSYCLTensor *tensor);

#endif
