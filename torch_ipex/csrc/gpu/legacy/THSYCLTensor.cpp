#include <legacy/THSYCLGeneral.h>
#include <legacy/THSYCLTensor.hpp>
#include <legacy/THSYCLTensorCopy.h>

#include <legacy/generic/THSYCLTensor.cpp>
#include <legacy/THSYCLGenerateAllTypes.h>

#include <legacy/generic/THSYCLTensor.cpp>
#include <legacy/THSYCLGenerateBoolType.h>

#include <core/TensorImplUtils.h>
#include <core/TensorInfo.h>

#include <functions/Resize.h>

int THSYCLTensor_nDimension(THSYCLState *state, const THSYCLTensor *self) {
  return THTensor_nDimension(self);
}

int THSYCLTensor_nDimensionLegacyNoScalars(THSYCLState *state, const THSYCLTensor *self) {
  return THTensor_nDimensionLegacyNoScalars(self);
}

int THSYCLTensor_nDimensionLegacyAll(THSYCLState *state, const THSYCLTensor *self) {
  return THTensor_nDimensionLegacyAll(self);
}

int64_t THSYCLTensor_size(THSYCLState *state, const THSYCLTensor *self, int dim) {
  THArgCheck((dim >= 0) && (dim < self->dim()), 2, "out of range");
  return self->size(dim);
}

int64_t THSYCLTensor_sizeLegacyNoScalars(THSYCLState *state, const THSYCLTensor *self, int dim) {
  return THTensor_sizeLegacyNoScalars(self, dim);
}

std::vector<int64_t> THSYCLTensor_sizesLegacyNoScalars(THSYCLState *state, const THSYCLTensor *self) {
  return THTensor_sizesLegacyNoScalars(self);
}

int64_t THSYCLTensor_stride(THSYCLState *state, const THSYCLTensor *self, int dim) {
  THArgCheck((dim >= 0) && (dim < self->dim()), 2, "out of range");
  return self->stride(dim);
}

int64_t THSYCLTensor_strideLegacyNoScalars(THSYCLState *state, const THSYCLTensor *self, int dim) {
  return THTensor_strideLegacyNoScalars(self, dim);
}

THSYCLTensor *THSYCLTensor_new(THSYCLState *state, caffe2::TypeMeta type_meta) {
  auto scalar_type = at::typeMetaToScalarType(type_meta);
  switch (scalar_type) {
    case at::ScalarType::Byte:
      return THSyclByteTensor_new(state);
    case at::ScalarType::Char:
      return THSyclCharTensor_new(state);
    case at::ScalarType::Short:
      return THSyclShortTensor_new(state);
    case at::ScalarType::Int:
      return THSyclIntTensor_new(state);
    case at::ScalarType::Long:
      return THSyclLongTensor_new(state);
    case at::ScalarType::Half:
      return THSyclHalfTensor_new(state);
    case at::ScalarType::Float:
      return THSyclTensor_new(state);
    case at::ScalarType::Double:
      return THSyclDoubleTensor_new(state);
    default:
       AT_ERROR("unexpected ScalarType: ", toString(scalar_type));
  }
}

void THSYCLTensor_resize(THSYCLState *state, THSYCLTensor *self, at::IntArrayRef size, at::IntArrayRef stride) {
  if(stride.data()) {
    THArgCheck(stride.size() == size.size(), 3, "invalid stride");
  }

#ifdef DEBUG
  THAssert(size.size() <= INT_MAX);
#endif
  THSYCLTensor_resizeNd(state, self, size.size(), size.data(), stride.data());
}

void THSYCLTensor_resizeAs(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src) {
  int isSame = 0;
  int d;
  if(self->dim() == src->dim())
  {
    isSame = 1;
    for(d = 0; d < self->dim(); d++)
    {
      if(self->size(d) != src->size(d))
      {
        isSame = 0;
        break;
      }
    }
  }

  if(!isSame)
    THSYCLTensor_resizeNd(state, self, src->dim(), THTensor_getSizePtr(src), NULL);
}

void THSYCLTensor_resizeNd(THSYCLState *state, THSYCLTensor *self, int nDimension, const int64_t *size, const int64_t *stride) {
  TORCH_CHECK(nDimension >= 0, "resizeNd nDimension must be non-negative");
  at::IntArrayRef sizes(size, nDimension);
  at::optional<at::IntArrayRef> strides;
  if (stride) {
    strides = at::IntArrayRef(stride, nDimension);
  }
  at::native::TensorImpl_resizeImpl(self, sizes, strides, /*device_guard=*/false);
}

void THSYCLTensor_set(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src) {
  if(self != src)
    THSYCLTensor_setStorageNd(state,
                           self,
                           THTensor_getStoragePtr(src),
                           src->storage_offset(),
                           src->dim(),
                           THTensor_getSizePtr(src),
                           THTensor_getStridePtr(src));
}

void THSYCLTensor_setStorage(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage_, ptrdiff_t storageOffset_, at::IntArrayRef size_, at::IntArrayRef stride_) {
  if (stride_.data()) {
    THArgCheck(size_.size() == stride_.size(), 5, "inconsistent size/stride sizes");
  }

  THSYCLTensor_setStorageNd(state,
                         self,
                         storage_,
                         storageOffset_,
                         size_.size(),
                         size_.data(),
                         stride_.data());
}

void THSYCLTensor_setStorageNd(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage, ptrdiff_t storageOffset, int nDimension, const int64_t *size, const int64_t *stride) {
  /* storage */
  if(THTensor_getStoragePtr(self) != storage)
  {
    if (!THTensor_getStoragePtr(self)) {
      THError("Tensor: invalid null storage");
    }
    auto data_type = THTensor_getStoragePtr(self)->dtype();
    if (storage) {
      at::raw::intrusive_ptr::incref(storage);
      THTensor_stealAndSetStoragePtr(self, storage);
    } else {
      THTensor_stealAndSetStoragePtr(self, THSYCLStorage_new(state, data_type));
    }
  }

  /* storageOffset */
  if (storageOffset < 0) {
    THError("Tensor: invalid storage offset");
  }
  self->set_storage_offset(storageOffset);

  /* size and stride */
  THSYCLTensor_resizeNd(state, self, nDimension, size, stride);
}

void THSYCLTensor_squeeze1d(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension) {
  int d;

  if(!src)
    src = self;

  THArgCheck(dimension < src->dim(), 3, "dimension out of range");

  THSYCLTensor_set(state, self, src);

  if(src->size(dimension) == 1)
  {
    for(d = dimension; d < self->dim()-1; d++)
    {
      self->set_size(d, self->size(d+1));
      self->set_stride(d, self->stride(d+1));
    }
    self->resize_dim((unsigned int)(self->dim() - 1));
  }
}

void THSYCLTensor_unsqueeze1d(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension) {
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension <= src->dim()), 3, "dimension out of range");

  THSYCLTensor_set(state, self, src);

  self->resize_dim(self->dim() + 1);
  for (d = self->dim()-1; d > dimension; d--) {
    self->set_size(d, self->size(d-1));
    self->set_stride(d, self->stride(d-1));
  }
  if (dimension+1 < self->dim()) {
    self->set_stride(dimension, self->size(dimension+1) * self->stride(dimension+1));
  } else {
    self->set_stride(dimension, 1);
  }
  self->set_size(dimension, 1);
}

bool THSYCLTensor_allContiguous(THSYCLState *state, THSYCLTensor **inputs, int numInputs) {
  THAssert(numInputs > 0);
  for (int i = 0; i < numInputs; ++i) {
    if (!inputs[i]->is_contiguous()) {
      return false;
    }
  }
  return true;
}

int64_t THSYCLTensor_nElement(THSYCLState *state, const THSYCLTensor *self) {
  if(THTensor_nDimensionLegacyAll(self) == 0) {
    return 0;
  } else {
    return self->numel();
  }
}

// NB: It is INVALID to call this on an UndefinedTensor
void THSYCLTensor_retain(THSYCLState *state, THSYCLTensor *self) {
  at::raw::intrusive_ptr::incref(self);
}

void THSYCLTensor_free(THSYCLState *state, THSYCLTensor *self) {
  THTensor_free(self);
}

int THSYCLTensor_getDevice(THSYCLState* state, const THSYCLTensor* tensor) {
  if (!THTensor_getStoragePtr(tensor)) return -1;
  return THSYCLStorage_getDevice(state, THTensor_getStoragePtr(tensor));
}

bool THSYCLTensor_allSameDevice(THSYCLState* state, THSYCLTensor ** inputs, int numInputs) {
  THAssert(numInputs > 0);
  int device = THSYCLTensor_getDevice(state, inputs[0]);
  for (int i = 1; i < numInputs; ++i) {
    if (THSYCLTensor_getDevice(state, inputs[i]) != device) {
      return false;
    }
  }
  return true;
}

bool THSYCLTensor_canUse32BitIndexMath(THSYCLState* state, const THSYCLTensor* t, ptrdiff_t max_elem) {
  ptrdiff_t elements = THSYCLTensor_nElement(state, t);
  if (elements >= max_elem) {
    return false;
  }
  if (t->dim() == 0) {
    return true;
  }

  ptrdiff_t offset = 0;
  ptrdiff_t linearId = elements - 1;

  for (int i = THSYCLTensor_nDimensionLegacyAll(state, t) - 1; i >= 0; --i) {
    ptrdiff_t curDimIndex =
      linearId % THSYCLTensor_size(state, t, i);
    ptrdiff_t curDimOffset = curDimIndex *
      THSYCLTensor_stride(state, t, i);
    offset += curDimOffset;
    linearId /= THSYCLTensor_size(state, t, i);
  }

  if (offset >= max_elem) {
    return false;
  }

  return true;
}

bool THSYCLTensor_all32BitIndexable(THSYCLState* state, THSYCLTensor** inputs, int numInputs) {
  for (int i = 0; i < numInputs; ++i) {
    if (!THSYCLTensor_canUse32BitIndexMath(state, inputs[i])) {
      return false;
    }
  }
  return true;
}

/* Due to the resize semantics of ops with `out=` keywords, if       */ \
/* the output `tensor` has the same shape as the output of the       */ \
/* reduction operation, then any noncontiguities in the output       */ \
/* `tensor` should be preserved. This needs to be special cased b/c  */ \
/* otherwise, when keepdim=False, the implementations of reduction   */ \
/* ops resize `tensor` to the reduced size with keepdim=True, and    */ \
/* then later squeeze `tensor` to the correct output size, breaking  */ \
/* the contiguity guarantees of the resize semantics.                */ \
void THSYCLTensor_preserveReduceDimSemantics(THSYCLState *state, THSYCLTensor *tensor,
                                          int in_dims, int64_t dimension, int keepdim) {
  int out_dims = THSYCLTensor_nDimensionLegacyAll(state, tensor);
  if (out_dims > 0 && !keepdim && out_dims == in_dims - 1) {
    THSYCLTensor_unsqueeze1d(state, tensor, tensor, dimension);
  }
}


namespace {

struct SizeAndStride {
  int64_t size;
  int64_t stride;
};

/*
 A comparator that will sort SizeAndStride structs by stride,
 in ascending order.
 */ 
int compareSizeAndStride(const void* a, const void* b) {
  const SizeAndStride* aS = (const SizeAndStride*) a;
  const SizeAndStride* bS = (const SizeAndStride*) b;
  
  if (aS->stride < bS->stride) return -1;
  if (aS->stride == bS->stride) return 0;
  return 1;
}

}

/* Returns false if there is no possibility that the tensor    */
/* has "overlapping" indices and true otherwise.               */
/* "Overlapping" indices are two+ valid indices that specify   */
/* the same offset within the tensor.                          */
/* The function does this by checking for a sufficient but not */
/* necessary condition of no overlap. In particular, that      */
/* that there exists an ordering of the tensor's dimensions    */
/* that is nicely "nested," with each dimension contained      */
/* within the next one.                                        */
bool THSYCLTensor_maybeOverlappingIndices(THSYCLState* state, const THSYCLTensor* t) {
  /* Extract size/stride arrays; only consider size >1 dims. */
  SizeAndStride info[MAX_SYCLTORCH_DIMS];

  int dims = THSYCLTensor_nDimensionLegacyAll(state, t);
  int nonSize1Dims = 0;
  for (int i = 0; i < dims; ++i) {
    int64_t size = THSYCLTensor_sizeLegacyNoScalars(state, t, i);

    if (size > 1) {
      info[nonSize1Dims].size = size;
      info[nonSize1Dims].stride =
        THSYCLTensor_stride(state, t, i);

      if (info[nonSize1Dims].stride < 1) {
        return true;
      }

      ++nonSize1Dims;
    }
  }

  /* Short-circuits if tensor is a single element.             */
  if (nonSize1Dims == 0) {
    return false;
  }

  /* Ascending order (innermost dimension in sorted view is at [0]) */
  qsort(info, nonSize1Dims, sizeof(SizeAndStride), compareSizeAndStride);

  for (int i = 0; i < (nonSize1Dims - 1); ++i) {
    if (((info[i].size - 1) * info[i].stride) >= info[i + 1].stride) {
      return true;
    }
  }

  return false;
}
