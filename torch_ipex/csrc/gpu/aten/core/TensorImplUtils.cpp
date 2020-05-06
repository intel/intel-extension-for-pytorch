#include <core/Guard.h>
#include <core/StorageImplUtils.h>
#include <core/TensorImplUtils.h>
#include <core/detail/TensorInfo.h>
#include <utils/General.h>

using namespace at::dpcpp::detail;

namespace at {
namespace dpcpp {

TensorImpl* TensorImpl_new(caffe2::TypeMeta type_meta) {
  return c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
             c10::intrusive_ptr<at::StorageImpl>::reclaim(
                 StorageImpl_new(type_meta)),
             c10::DispatchKey::DPCPPTensorId)
      .release();
}

at::Tensor TensorImpl_wrap(TensorImpl* tensor) {
  c10::raw::intrusive_ptr::incref(tensor);
  return at::Tensor(c10::intrusive_ptr<at::TensorImpl>::reclaim(tensor));
}

int TensorImpl_nDimension(const at::TensorImpl* self) {
  return self->dim();
}

int TensorImpl_nDimensionLegacyNoScalars(const at::TensorImpl* self) {
  if (self->dim() == 0) {
    return 1;
  } else {
    return self->dim();
  }
}

int TensorImpl_nDimensionLegacyAll(const at::TensorImpl* self) {
  if (self->is_empty()) {
    return 0;
  } else if (self->dim() == 0) {
    return 1;
  } else {
    return self->dim();
  }
}

const int64_t* TensorImpl_getSizePtr(at::TensorImpl* self) {
  return self->sizes().data();
}

int64_t TensorImpl_size(const at::TensorImpl* self, int dim) {
  TORCH_CHECK((dim >= 0) && (dim < self->dim()), "out of range");
  return self->size(dim);
}

int64_t TensorImpl_sizeLegacyNoScalars(const at::TensorImpl* self, int dim) {
  TORCH_CHECK(
      (dim >= 0) && (dim < TensorImpl_nDimensionLegacyNoScalars(self)),
      "dimension %d out of range of %dD tensor",
      dim,
      TensorImpl_nDimensionLegacyNoScalars(self));
  return self->dim() == 0 ? 1 : self->size(dim);
}

std::vector<int64_t> TensorImpl_sizesLegacyNoScalars(
    const at::TensorImpl* self) {
  if (self->dim() == 0) {
    return {1};
  } else {
    return self->sizes().vec();
  }
}

const int64_t* TensorImpl_getStridePtr(at::TensorImpl* self) {
  return self->strides().data();
}

int64_t TensorImpl_stride(const at::TensorImpl* self, int dim) {
  TORCH_CHECK((dim >= 0) && (dim < self->dim()), "out of range");
  return self->stride(dim);
}

int64_t TensorImpl_strideLegacyNoScalars(const at::TensorImpl* self, int dim) {
  TORCH_CHECK(
      (dim >= 0) && (dim < TensorImpl_nDimensionLegacyNoScalars(self)),
      "dimension %d out of range of %dD tensor",
      dim,
      TensorImpl_nDimensionLegacyNoScalars(self));
  return self->dim() == 0 ? 1 : self->stride(dim);
}

TensorImpl* TensorImpl_resizeImpl(
    at::TensorImpl* self,
    at::IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool device_guard) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  // NB: We don't need to hold device guard when calling from TH
  OptionalDPCPPGuard guard;
  if (device_guard) {
    guard.set_index(self->storage().device().index());
  }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel
    for (size_t dim = 0; dim < size.size(); ++dim) {
      // FIXME:  Don't rely on storage_size being negative because thi
      // may not be true for some edge cases.
      if (size[dim] == 0) {
        storage_size = 0;
        break;
      }
      storage_size += (size[dim] - 1) * stride.value()[dim];
    }
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }

  // maybe resize storage
  if (storage_size + self->storage_offset() > 0) {
    if (!TensorImpl_getStoragePtr(self)) {
      TORCH_CHECK(0, "Tensor: invalid null storage");
    }
    if (storage_size + self->storage_offset() > self->storage().numel()) {
      StorageImpl_resize(
          TensorImpl_getStoragePtr(self),
          storage_size + self->storage_offset());
    }
  }
  return self;
}

void TensorImpl_resize(
    at::TensorImpl* self,
    at::IntArrayRef size,
    at::IntArrayRef stride) {
  if (stride.data()) {
    TORCH_CHECK(stride.size() == size.size(), "invalid stride");
  }

#ifdef DEBUG
  TORCH_INTERNAL_ASSERT(size.size() <= INT_MAX);
#endif
  TensorImpl_resizeNd(self, size.size(), size.data(), stride.data());
}

void TensorImpl_resizeAs(at::TensorImpl* self, TensorImpl* src) {
  int isSame = 0;
  int d;
  if (self->dim() == src->dim()) {
    isSame = 1;
    for (d = 0; d < self->dim(); d++) {
      if (self->size(d) != src->size(d)) {
        isSame = 0;
        break;
      }
    }
  }

  if (!isSame)
    TensorImpl_resizeNd(self, src->dim(), TensorImpl_getSizePtr(src), NULL);
}

void TensorImpl_resizeNd(
    at::TensorImpl* self,
    int nDimension,
    const int64_t* size,
    const int64_t* stride) {
  TORCH_CHECK(nDimension >= 0, "resizeNd nDimension must be non-negative");
  at::IntArrayRef sizes(size, nDimension);
  at::optional<at::IntArrayRef> strides;
  if (stride) {
    strides = at::IntArrayRef(stride, nDimension);
  }
  TensorImpl_resizeImpl(self, sizes, strides, /*device_guard=*/false);
}

at::StorageImpl* TensorImpl_getStoragePtr(const at::TensorImpl* tensor) {
  // Within PyTorch, the invariant is that storage_ is always
  // initialized; we never have tensors that don't have any storage.
  // However, for Caffe2, this is not true, because they have permitted
  // tensors to be allocated without specifying what scalar type
  // they should be, only to be filled when GetMutableData is called
  // for the first time (providing the necessary type).  It is an ERROR to
  // invoke any PyTorch operations on such a half-constructed storage,
  // and this check tests for that case.
  TORCH_CHECK(
      tensor->storage(),
      "Cannot use PyTorch operations on a half-constructed "
      "tensor.  If this tensor came from Caffe2, please call GetMutableData on "
      "it first; otherwise, this is a bug, please report it.");
  return tensor->storage().unsafeGetStorageImpl();
}

// NB: Steals ownership of storage
void TensorImpl_stealAndSetStoragePtr(
    at::TensorImpl* tensor,
    at::StorageImpl* storage) {
  // Caffe2 might have tensors whose storages are null, but we
  // don't allow it in PyTorch.
  TORCH_INTERNAL_ASSERT(storage);
  // Caffe2 also has uninitialized dtype states, which we disallow here
  TORCH_INTERNAL_ASSERT(tensor->storage().dtype() == storage->dtype());

  // We used to allow this, but this breaks device caching.
  // Let's put an actual error message for this one.
  TORCH_CHECK(
      tensor->storage().device() == storage->device(),
      "Attempted to set the storage of a tensor on device \"",
      tensor->storage().device(),
      "\" to a storage on different device \"",
      storage->device(),
      "\".  This is no longer allowed; the devices must match.");
  tensor->set_storage(
      at::Storage(c10::intrusive_ptr<at::StorageImpl>::reclaim(storage)));
}

void TensorImpl_set(at::TensorImpl* self, at::TensorImpl* src) {
  if (self != src)
    TensorImpl_setStorageNd(
        self,
        TensorImpl_getStoragePtr(src),
        src->storage_offset(),
        src->dim(),
        TensorImpl_getSizePtr(src),
        TensorImpl_getStridePtr(src));
}

void TensorImpl_setStorage(
    at::TensorImpl* self,
    at::StorageImpl* storage_,
    ptrdiff_t storageOffset_,
    at::IntArrayRef size_,
    at::IntArrayRef stride_) {
  if (stride_.data()) {
    TORCH_CHECK(
        size_.size() == stride_.size(), "inconsistent size/stride sizes");
  }

  TensorImpl_setStorageNd(
      self,
      storage_,
      storageOffset_,
      size_.size(),
      size_.data(),
      stride_.data());
}

void TensorImpl_setStorageNd(
    at::TensorImpl* self,
    at::StorageImpl* storage,
    ptrdiff_t storageOffset,
    int nDimension,
    const int64_t* size,
    const int64_t* stride) {
  /* storage */
  if (TensorImpl_getStoragePtr(self) != storage) {
    if (!TensorImpl_getStoragePtr(self)) {
      TORCH_CHECK(false, "Tensor: invalid null storage");
    }
    auto data_type = TensorImpl_getStoragePtr(self)->dtype();
    if (storage) {
      at::raw::intrusive_ptr::incref(storage);
      TensorImpl_stealAndSetStoragePtr(self, storage);
    } else {
      TensorImpl_stealAndSetStoragePtr(self, StorageImpl_new(data_type));
    }
  }

  /* storageOffset */
  if (storageOffset < 0) {
    TORCH_CHECK(false, "Tensor: invalid storage offset");
  }
  self->set_storage_offset(storageOffset);

  /* size and stride */
  TensorImpl_resizeNd(self, nDimension, size, stride);
}

bool TensorImpl_isSetTo(const at::TensorImpl* self, const at::TensorImpl* src) {
  if (TensorImpl_getStoragePtr(self) == TensorImpl_getStoragePtr(src) &&
      self->storage_offset() == src->storage_offset() &&
      self->dim() == src->dim()) {
    int d;
    for (d = 0; d < self->dim(); ++d) {
      if (self->size(d) != src->size(d) || self->stride(d) != src->stride(d))
        return false;
    }
    return true;
  }
  return false;
}

void TensorImpl_squeeze1d(
    at::TensorImpl* self,
    at::TensorImpl* src,
    int dimension) {
  int d;

  if (!src)
    src = self;

  TORCH_CHECK(dimension < src->dim(), "dimension out of range");

  TensorImpl_set(self, src);

  if (src->size(dimension) == 1) {
    at::DimVector newSize(static_cast<size_t>(self->dim() - 1));
    at::DimVector newStride(static_cast<size_t>(self->dim() - 1));
    for (d = 0; d < dimension; d++) {
      newSize[d] = self->size(d);
      newStride[d] = self->stride(d);
    }

    for (d = dimension; d < self->dim() - 1; d++) {
      newSize[d] = self->size(d + 1);
      newStride[d] = self->stride(d + 1);
    }
    self->set_sizes_and_strides(newSize, newStride);
  }
}

void TensorImpl_unsqueeze1d(
    at::TensorImpl* self,
    at::TensorImpl* src,
    int dimension) {
  int d;

  if (!src)
    src = self;

  TORCH_CHECK(
      (dimension >= 0) && (dimension <= src->dim()), "dimension out of range");

  TensorImpl_set(self, src);

  at::DimVector newSize(static_cast<size_t>(/* size */ self->dim() + 1));
  at::DimVector newStride(static_cast<size_t>(/* size */ self->dim() + 1));

  for (d = self->dim(); d > dimension; d--) {
    newSize[d] = self->size(d - 1);
    newStride[d] = self->stride(d - 1);
  }
  if (dimension < self->dim()) {
    newStride[dimension] = self->size(dimension) * self->stride(dimension);
  } else {
    newStride[dimension] = 1;
  }
  newSize[dimension] = 1;
  for (d = dimension - 1; d >= 0; d--) {
    newSize[d] = self->size(d);
    newStride[d] = self->stride(d);
  }
  self->set_sizes_and_strides(newSize, newStride);
}

bool TensorImpl_allContiguous(at::TensorImpl** inputs, int numInputs) {
  TORCH_INTERNAL_ASSERT(numInputs > 0);
  for (int i = 0; i < numInputs; ++i) {
    if (!inputs[i]->is_contiguous()) {
      return false;
    }
  }
  return true;
}

int64_t TensorImpl_nElement(const at::TensorImpl* self) {
  if (TensorImpl_nDimensionLegacyAll(self) == 0) {
    return 0;
  } else {
    return self->numel();
  }
}

// NB: It is INVALID to call this on an UndefinedTensor
void TensorImpl_retain(at::TensorImpl* self) {
  at::raw::intrusive_ptr::incref(self);
}

void TensorImpl_free(at::TensorImpl* self) {
  if (!self)
    return;
  c10::raw::intrusive_ptr::decref(self);
}

int TensorImpl_getDevice(const at::TensorImpl* tensor) {
  if (!TensorImpl_getStoragePtr(tensor))
    return -1;
  return StorageImpl_getDevice(TensorImpl_getStoragePtr(tensor));
}

bool TensorImpl_allSameDevice(at::TensorImpl** inputs, int numInputs) {
  TORCH_INTERNAL_ASSERT(numInputs > 0);
  int device = TensorImpl_getDevice(inputs[0]);
  for (int i = 1; i < numInputs; ++i) {
    if (TensorImpl_getDevice(inputs[i]) != device) {
      return false;
    }
  }
  return true;
}

bool TensorImpl_canUse32BitIndexMath(
    const at::TensorImpl* t,
    ptrdiff_t max_elem) {
  ptrdiff_t elements = TensorImpl_nElement(t);
  if (elements >= max_elem) {
    return false;
  }
  if (t->dim() == 0) {
    return true;
  }

  ptrdiff_t offset = 0;
  ptrdiff_t linearId = elements - 1;

  for (int i = TensorImpl_nDimensionLegacyAll(t) - 1; i >= 0; --i) {
    ptrdiff_t curDimIndex = linearId % TensorImpl_size(t, i);
    ptrdiff_t curDimOffset = curDimIndex * TensorImpl_stride(t, i);
    offset += curDimOffset;
    linearId /= TensorImpl_size(t, i);
  }

  if (offset >= max_elem) {
    return false;
  }

  return true;
}

bool TensorImpl_all32BitIndexable(at::TensorImpl** inputs, int numInputs) {
  for (int i = 0; i < numInputs; ++i) {
    if (!TensorImpl_canUse32BitIndexMath(inputs[i])) {
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
void TensorImpl_preserveReduceDimSemantics(TensorImpl *tensor,
                                          int in_dims, int64_t dimension, int keepdim) {
  int out_dims = TensorImpl_nDimensionLegacyAll(tensor);
  if (out_dims > 0 && !keepdim && out_dims == in_dims - 1) {
    TensorImpl_unsqueeze1d(tensor, tensor, dimension);
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
  const SizeAndStride* aS = (const SizeAndStride*)a;
  const SizeAndStride* bS = (const SizeAndStride*)b;

  if (aS->stride < bS->stride)
    return -1;
  if (aS->stride == bS->stride)
    return 0;
  return 1;
}
} // namespace

/* Returns false if there is no possibility that the tensor    */
/* has "overlapping" indices and true otherwise.               */
/* "Overlapping" indices are two+ valid indices that specify   */
/* the same offset within the tensor.                          */
/* The function does this by checking for a sufficient but not */
/* necessary condition of no overlap. In particular, that      */
/* that there exists an ordering of the tensor's dimensions    */
/* that is nicely "nested," with each dimension contained      */
/* within the next one.                                        */
bool TensorImpl_maybeOverlappingIndices(const at::TensorImpl* t) {
  /* Extract size/stride arrays; only consider size >1 dims. */
  SizeAndStride info[MAX_DPCPPTORCH_DIMS];

  int dims = TensorImpl_nDimensionLegacyAll(t);
  int nonSize1Dims = 0;
  for (int i = 0; i < dims; ++i) {
    int64_t size = TensorImpl_sizeLegacyNoScalars(t, i);

    if (size > 1) {
      info[nonSize1Dims].size = size;
      info[nonSize1Dims].stride = TensorImpl_stride(t, i);

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

DPCPPDescBuff TensorImpl_sizeDesc(const at::TensorImpl* tensor) {
  const int L = DPCPP_DESC_BUFF_LEN;
  DPCPPDescBuff buf;
  char* str = buf.str;
  int n = 0;
  n += snprintf(str, L - n, "[");
  int i;
  for (i = 0; i < tensor->dim(); i++) {
    if (n >= L)
      break;
    n += snprintf(str + n, L - n, "%" PRId64, tensor->size(i));
    if (i < tensor->dim() - 1) {
      n += snprintf(str + n, L - n, " x ");
    }
  }
  if (n < L - 2) {
    snprintf(str + n, L - n, "]");
  } else {
    snprintf(str + L - 5, 5, "...]");
  }
  return buf;
}

at::TensorImpl* TensorImpl_Unwrap(const at::Tensor& tensor) {
  return tensor.unsafeGetTensorImpl();
}

} // namespace dpcpp
} // namespace at
