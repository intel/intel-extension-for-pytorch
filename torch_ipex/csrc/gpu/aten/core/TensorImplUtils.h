#pragma once

#include <c10/core/TensorImpl.h>
#include <c10/core/StorageImpl.h>

namespace at { namespace native {

#define SYCL_DESC_BUFF_LEN 64

typedef struct SYCLDescBuff
{
  char str[SYCL_DESC_BUFF_LEN];
} SYCLDescBuff;

TensorImpl *TensorImpl_new();
at::Tensor TensorImpl_wrap(TensorImpl *tensor);
int TensorImpl_nDimension(const at::TensorImpl *self);
int TensorImpl_nDimensionLegacyNoScalars(const at::TensorImpl *self);
int TensorImpl_nDimensionLegacyAll(const at::TensorImpl *self);
const int64_t* TensorImpl_getSizePtr(at::TensorImpl* tensor);
int64_t TensorImpl_size(const at::TensorImpl *self, int dim);
int64_t TensorImpl_sizeLegacyNoScalars(const at::TensorImpl *self, int dim);
std::vector<int64_t> TensorImpl_sizesLegacyNoScalars(const at::TensorImpl *self);
const int64_t* TensorImpl_getStridePtr(at::TensorImpl* tensor);
int64_t TensorImpl_stride(const at::TensorImpl *self, int dim);
int64_t TensorImpl_strideLegacyNoScalars(const at::TensorImpl *self, int dim);
// THSYCL_API THSYCLTensor *THSYCLTensor_new(THSYCLState *state, caffe2::TypeMeta type_meta);
TensorImpl* TensorImpl_resizeImpl(at::TensorImpl* self, at::IntArrayRef size, c10::optional<IntArrayRef> stride, bool device_guard = true);
void TensorImpl_resize(at::TensorImpl *self, at::IntArrayRef size, at::IntArrayRef stride);
void TensorImpl_resizeAs(at::TensorImpl *self, TensorImpl *src);
void TensorImpl_resizeNd(at::TensorImpl *self, int nDimension, const int64_t *size, const int64_t *stride);
at::StorageImpl* TensorImpl_getStoragePtr(const at::TensorImpl* tensor);
void TensorImpl_stealAndSetStoragePtr(at::TensorImpl* tensor, at::StorageImpl* storage);
void TensorImpl_set(at::TensorImpl *self, at::TensorImpl *src);
void TensorImpl_setStorage(at::TensorImpl *self, at::StorageImpl *storage_, ptrdiff_t storageOffset_, at::IntArrayRef size_, at::IntArrayRef stride_);
void TensorImpl_setStorageNd(at::TensorImpl *self, at::StorageImpl *storage, ptrdiff_t storageOffset, int nDimension, const int64_t *size, const int64_t *stride);
void TensorImpl_squeeze1d(at::TensorImpl *self, at::TensorImpl *src, int dimension);
void TensorImpl_unsqueeze1d(at::TensorImpl *self, at::TensorImpl *src, int dimension);
bool TensorImpl_allContiguous(at::TensorImpl **inputs, int numInputs);
int64_t TensorImpl_nElement(const at::TensorImpl *self);
void TensorImpl_retain(at::TensorImpl *self);
void TensorImpl_free(at::TensorImpl *self);
int TensorImpl_getDevice(const at::TensorImpl* tensor);
bool TensorImpl_allSameDevice(at::TensorImpl ** inputs, int numInputs);
bool TensorImpl_canUse32BitIndexMath(const at::TensorImpl* t, ptrdiff_t max_elem=INT32_MAX);
bool TensorImpl_all32BitIndexable(at::TensorImpl** inputs, int numInputs);
void TensorImpl_preserveReduceDimSemantics(TensorImpl *tensor, int in_dims, int64_t dimension, int keepdim);
bool TensorImpl_maybeOverlappingIndices(const at::TensorImpl* t);

SYCLDescBuff TensorImpl_sizeDesc(const at::TensorImpl *tensor);

} // native::
} // at::
