#pragma once

// STOP!!! Thinking of including this header directly?  Please
// // read Note [TH abstraction violation]

#include <THDP/THSYCLTensor.h>
#include <TH/THTensor.hpp>
#include <THDP/THSYCLStorage.hpp>
#include <THDP/THSYCLGeneral.h>

#include <ATen/ATen.h>

// See [NOTE: nDimension vs nDimensionLegacyNoScalars vs nDimensionLegacyAll]
THSYCL_API int THSYCLTensor_nDimension(THSYCLState *state, const THSYCLTensor *self);
THSYCL_API int THSYCLTensor_nDimensionLegacyNoScalars(THSYCLState *state, const THSYCLTensor *self);
THSYCL_API int THSYCLTensor_nDimensionLegacyAll(THSYCLState *state, const THSYCLTensor *self);

THSYCL_API int64_t THSYCLTensor_size(THSYCLState *state, const THSYCLTensor *self, int dim);
THSYCL_API int64_t THSYCLTensor_sizeLegacyNoScalars(THSYCLState *state, const THSYCLTensor *self, int dim);
THSYCL_API int64_t THSYCLTensor_stride(THSYCLState *state, const THSYCLTensor *self, int dim);
THSYCL_API int64_t THSYCLTensor_strideLegacyNoScalars(THSYCLState *state, const THSYCLTensor *self, int dim);

std::vector<int64_t> THSYCLTensor_sizesLegacyNoScalars(THSYCLState *state, const THSYCLTensor *self);

#include <THDP/generic/THSYCLTensorFastGetSet.hpp>
#include <THDP/THSYCLGenerateAllTypes.h>


THSYCL_API THSYCLTensor *THSYCLTensor_new(THSYCLState *state, caffe2::TypeMeta type_meta);

THSYCL_API void THSYCLTensor_resize(THSYCLState *state, THSYCLTensor *tensor, at::IntArrayRef size, at::IntArrayRef stride);
THSYCL_API void THSYCLTensor_resizeNd(THSYCLState *state, THSYCLTensor *tensor, int nDimension, const int64_t *size, const int64_t *stride);
THSYCL_API void THSYCLTensor_resizeAs(THSYCLState *state, THSYCLTensor *tensor, THSYCLTensor *src);

THSYCL_API void THSYCLTensor_set(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src);
THSYCL_API void THSYCLTensor_setStorage(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage_, ptrdiff_t storageOffset_, at::IntArrayRef size_, at::IntArrayRef stride_);
THSYCL_API void THSYCLTensor_setStorageNd(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage, ptrdiff_t storageOffset, int nDimension, const int64_t *size, const int64_t *stride);

THSYCL_API void THSYCLTensor_squeeze1d(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension_);
THSYCL_API void THSYCLTensor_unsqueeze1d(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension_);

THSYCL_API bool THSYCLTensor_allContiguous(THSYCLState *state, THSYCLTensor **inputs, int numInputs);
THSYCL_API ptrdiff_t THSYCLTensor_nElement(THSYCLState *state, const THSYCLTensor *self);

THSYCL_API void THSYCLTensor_retain(THSYCLState *state, THSYCLTensor *self);
THSYCL_API void THSYCLTensor_free(THSYCLState *state, THSYCLTensor *self);

THSYCL_API int THSYCLTensor_getDevice(THSYCLState* state, const THSYCLTensor* tensor);
THSYCL_API bool THSYCLTensor_allSameDevice(THSYCLState* state, THSYCLTensor ** inputs, int numInputs);

/* Can we use 32 bit math for indexing? */
THSYCL_API bool THSYCLTensor_canUse32BitIndexMath(THSYCLState* state, const THSYCLTensor* t, ptrdiff_t max_elem=INT32_MAX);
/* Are all tensors 32-bit indexable? */
THSYCL_API bool THSYCLTensor_all32BitIndexable(THSYCLState* state, THSYCLTensor** inputs, int numInputs);
THSYCL_API void THSYCLTensor_preserveReduceDimSemantics(THSYCLState *state, THSYCLTensor *tensor, int in_dims,
                                                  int64_t dimension, int keepdim);
/* Returns false if there is no possibility that the tensor    */
/* has more than one index that references the same datapoint, */
/* true otherwise.                                             */
THSYCL_API bool THSYCLTensor_maybeOverlappingIndices(THSYCLState* state, const THSYCLTensor* t);

#include <THDP/generic/THSYCLTensor.hpp>
#include <THDP/THSYCLGenerateAllTypes.h>

#include <THDP/generic/THSYCLTensor.hpp>
#include <THDP/THSYCLGenerateBoolType.h>
