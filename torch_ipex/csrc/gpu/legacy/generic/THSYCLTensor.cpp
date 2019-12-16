#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy/generic/THSYCLTensor.cpp"
#else

#include <ATen/InferSize.h>
#include <ATen/TensorUtils.h>
#include <core/SYCLException.h>

#include <aten_ipex_tensor_type.h>


/**** access methods ****/
THSYCLStorage *THSYCLTensor_(storage)(THSYCLState *state, const THSYCLTensor *self)
{
    return THTensor_getStoragePtr(self);
}

ptrdiff_t THSYCLTensor_(storageOffset)(THSYCLState *state, const THSYCLTensor *self)
{
    return self->storage_offset();
}

int THSYCLTensor_(nDimension)(THSYCLState *state, const THSYCLTensor *self)
{
    return THSYCLTensor_nDimension(state, self);
}

int THSYCLTensor_(nDimensionLegacyNoScalars)(THSYCLState *state, const THSYCLTensor *self)
{
    return THSYCLTensor_nDimensionLegacyNoScalars(state, self);
}

int THSYCLTensor_(nDimensionLegacyAll)(THSYCLState *state, const THSYCLTensor *self)
{
    return THSYCLTensor_nDimensionLegacyAll(state, self);
}

int64_t THSYCLTensor_(size)(THSYCLState *state, const THSYCLTensor *self, int dim)
{
    return THSYCLTensor_size(state, self, dim);
}

int64_t THSYCLTensor_(sizeLegacyNoScalars)(THSYCLState *state, const THSYCLTensor *self, int dim)
{
    return THTensor_sizeLegacyNoScalars(self, dim);
}

int64_t THSYCLTensor_(stride)(THSYCLState *state, const THSYCLTensor *self, int dim)
{
    return THSYCLTensor_stride(state, self, dim);
}

int64_t THSYCLTensor_(strideLegacyNoScalars)(THSYCLState *state, const THSYCLTensor *self, int dim)
{
    return THTensor_strideLegacyNoScalars(self, dim);
}

scalar_t *THSYCLTensor_(data)(THSYCLState *state, const THSYCLTensor *self)
{
  if(THTensor_getStoragePtr(self))
    return (THSYCLStorage_(data)(state, THTensor_getStoragePtr(self))+self->storage_offset());
  else
    return NULL;
}

/**** creation methods ****/

/* Empty init */
THSYCLTensor *THSYCLTensor_(new)(THSYCLState *state)
{
  return c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
    c10::intrusive_ptr<at::StorageImpl>::reclaim(THSYCLStorage_(new)(state)),
    at::torch_ipex::DPCPPTensorId()
  ).release();
}


/* Pointer-copy init */
THSYCLTensor *THSYCLTensor_(newWithTensor)(THSYCLState *state, THSYCLTensor *tensor)
{
  THSYCLTensor *self = c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
    c10::intrusive_ptr<at::StorageImpl>::reclaim(THSYCLStorage_(new)(state)),
    at::torch_ipex::DPCPPTensorId()
  ).release();
  THSYCLTensor_(setStorageNd)(state,
                              self,
                              THTensor_getStoragePtr(tensor),
                              tensor->storage_offset(),
                              tensor->dim(),
                              THTensor_getSizePtr(tensor),
                              THTensor_getStridePtr(tensor));
  return self;
}

/* Storage init */
THSYCLTensor *THSYCLTensor_(newWithStorage)(THSYCLState *state, THSYCLStorage *storage, ptrdiff_t storageOffset, c10::IntArrayRef sizes, c10::IntArrayRef strides) {
  if (strides.data()) {
    TORCH_CHECK(sizes.size() == strides.size(), "number of sizes and strides must match");
  }
  THSYCLTensor *self = c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
    c10::intrusive_ptr<at::StorageImpl>::reclaim(THSYCLStorage_(new)(state)),
    at::torch_ipex::DPCPPTensorId()
  ).release();
  THSYCLTensor_(setStorageNd)(state, self, storage, storageOffset, sizes.size(),
                           const_cast<int64_t*>(sizes.data()), const_cast<int64_t*>(strides.data()));

  return self;
}

THSYCLTensor *THSYCLTensor_(newWithStorage1d)(THSYCLState *state, THSYCLStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0)
{
  return THSYCLTensor_(newWithStorage)(state, storage, storageOffset, {size0}, {stride0});
}

THSYCLTensor *THSYCLTensor_(newWithStorage2d)(THSYCLState *state, THSYCLStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1)
{
  return THSYCLTensor_(newWithStorage)(state, storage, storageOffset, {size0, size1}, {stride0, stride1});
}

THSYCLTensor *THSYCLTensor_(newWithStorage3d)(THSYCLState *state, THSYCLStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2)
{
  return THSYCLTensor_(newWithStorage)(state, storage, storageOffset, {size0, size1, size2}, {stride0, stride1, stride2});
}

THSYCLTensor *THSYCLTensor_(newWithStorage4d)(THSYCLState *state, THSYCLStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2,
                               int64_t size3, int64_t stride3)
{
  return THSYCLTensor_(newWithStorage)(state, storage, storageOffset,
                                            {size0, size1, size2, size3},
                                            {stride0, stride1, stride2, stride3});
}

THSYCLTensor *THSYCLTensor_(newWithSize)(THSYCLState *state, at::IntArrayRef size, at::IntArrayRef stride)
{
  return THSYCLTensor_(newWithStorage)(state, NULL, 0, size, stride);
}

THSYCLTensor *THSYCLTensor_(newWithSize1d)(THSYCLState *state, int64_t size0)
{
  return THSYCLTensor_(newWithSize)(state, {size0}, {});
}

THSYCLTensor *THSYCLTensor_(newWithSize2d)(THSYCLState *state, int64_t size0, int64_t size1)
{
  return THSYCLTensor_(newWithSize)(state, {size0, size1}, {});
}

THSYCLTensor *THSYCLTensor_(newWithSize3d)(THSYCLState *state, int64_t size0, int64_t size1, int64_t size2)
{
  return THSYCLTensor_(newWithSize)(state, {size0, size1, size2}, {});
}

THSYCLTensor *THSYCLTensor_(newWithSize4d)(THSYCLState *state, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  return THSYCLTensor_(newWithSize)(state, {size0, size1, size2, size3}, {});
}

THSYCLTensor *THSYCLTensor_(newClone)(THSYCLState *state, THSYCLTensor *self)
{
  THSYCLTensor *tensor = THSYCLTensor_(new)(state);
  THSYCLTensor_(resizeAs)(state, tensor, self);
  THSYCLTensor_(copy)(state, tensor, self);
  return tensor;
}

THSYCLTensor *THSYCLTensor_(newContiguous)(THSYCLState *state, THSYCLTensor *self)
{
  if(!THSYCLTensor_(isContiguous)(state, self)) {
    return THSYCLTensor_(newClone)(state, self);
  } else {
    THSYCLTensor_(retain)(state, self);
    return self;
  }
}

THSYCLTensor *THSYCLTensor_(newSelect)(THSYCLState *state, THSYCLTensor *tensor, int dimension_, int64_t sliceIndex_)
{
  THSYCLTensor *self = THSYCLTensor_(newWithTensor)(state, tensor);
  THSYCLTensor_(select)(state, self, NULL, dimension_, sliceIndex_);
  return self;
}

THSYCLTensor *THSYCLTensor_(newNarrow)(THSYCLState *state, THSYCLTensor *tensor, int dimension_, int64_t firstIndex_, int64_t size_)
{
  THSYCLTensor *self = THSYCLTensor_(newWithTensor)(state, tensor);
  THSYCLTensor_(narrow)(state, self, NULL, dimension_, firstIndex_, size_);
  return self;
}

THSYCLTensor *THSYCLTensor_(newTranspose)(THSYCLState *state, THSYCLTensor *tensor, int dimension1_, int dimension2_)
{
  THSYCLTensor *self = THSYCLTensor_(newWithTensor)(state, tensor);
  THSYCLTensor_(transpose)(state, self, NULL, dimension1_, dimension2_);
  return self;
}

THSYCLTensor *THSYCLTensor_(newUnfold)(THSYCLState *state, THSYCLTensor *tensor, int dimension_, int64_t size_, int64_t step_)
{
  THSYCLTensor *self = THSYCLTensor_(newWithTensor)(state, tensor);
  THSYCLTensor_(unfold)(state, self, NULL, dimension_, size_, step_);
  return self;
}

THSYCLTensor *THSYCLTensor_(newView)(THSYCLState *state, THSYCLTensor *tensor, at::IntArrayRef size)
{
  ptrdiff_t numel = THSYCLTensor_(nElement)(state, tensor);
  THSYCLTensor *self = THSYCLTensor_(new)(state);
  auto inferred_size = at::infer_size(size, numel);
  auto stride = at::detail::computeStride(tensor->sizes(),
                                          tensor->strides(),
                                          inferred_size);
  THArgCheck(stride.has_value(), 2, "view size is "
    "not compatible with input tensor's size and stride (at least one dimension spans "
    "across two contiguous subspaces). Call .contiguous() before .view().");
  auto stride_value = *stride;
  THSYCLTensor_setStorage(state, self, THTensor_getStoragePtr(tensor), tensor->storage_offset(), inferred_size, stride_value);
  return self;
}

// Collapses the first two dimensions of a tensor.
// Assumes the input tensor is contiguous.
THSYCLTensor *THSYCLTensor_(newFoldBatchDim)(THSYCLState *state, THSYCLTensor *input) {
  int in_dims = THSYCLTensor_(nDimensionLegacyAll)(state, input);
  THArgCheck(in_dims >= 2, 1, "Tensor needs to have at least two dimensions");
  THArgCheck(THSYCLTensor_(isContiguous)(state, input), 1,
             "Tensor must be contiguous");
  std::vector<int64_t> new_size(in_dims - 1);
  new_size[0] = THSYCLTensor_(size)(state, input, 0) * THSYCLTensor_(size)(state, input, 1);
  for (int i = 2; i < in_dims; i++) {
    new_size[i - 1] = THSYCLTensor_(size)(state, input, i);
  }
  THSYCLTensor *output = THSYCLTensor_(newView)(state, input, new_size);
  return output;
}


/* Resize */
void THSYCLTensor_(resize)(THSYCLState *state, THSYCLTensor *self, at::IntArrayRef size, at::IntArrayRef stride)
{
  THSYCLTensor_resize(state, self, size, stride);
}

void THSYCLTensor_(resizeAs)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src)
{
  THSYCLTensor_resizeAs(state, self, src);
}

void THSYCLTensor_(resize0d)(THSYCLState *state, THSYCLTensor *tensor)
{
  THSYCLTensor_resizeNd(state, tensor, 0, {}, nullptr);
}

void THSYCLTensor_(resize1d)(THSYCLState *state, THSYCLTensor *tensor, int64_t size0)
{
  int64_t size[1] = {size0};
  THSYCLTensor_resizeNd(state, tensor, 1, size, nullptr);
}

void THSYCLTensor_(resize2d)(THSYCLState *state, THSYCLTensor *tensor, int64_t size0, int64_t size1)
{
  int64_t size[2] = {size0, size1};
  THSYCLTensor_resizeNd(state, tensor, 2, size, nullptr);
}

void THSYCLTensor_(resize3d)(THSYCLState *state, THSYCLTensor *tensor, int64_t size0, int64_t size1, int64_t size2)
{
  int64_t size[3] = {size0, size1, size2};
  THSYCLTensor_resizeNd(state, tensor, 3, size, nullptr);
}

void THSYCLTensor_(resize4d)(THSYCLState *state, THSYCLTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};
  THSYCLTensor_resizeNd(state, self, 4, size, nullptr);
}

void THSYCLTensor_(resize5d)(THSYCLState *state, THSYCLTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4)
{
  int64_t size[5] = {size0, size1, size2, size3, size4};
  THSYCLTensor_resizeNd(state, self, 5, size, nullptr);
}

void THSYCLTensor_(set)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src)
{
  THSYCLTensor_set(state, self, src);
}

void THSYCLTensor_(setStorage)(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage_, ptrdiff_t storageOffset_, at::IntArrayRef size_, at::IntArrayRef stride_) {
  THSYCLTensor_setStorage(state, self, storage_, storageOffset_, size_, stride_);
}

void THSYCLTensor_(setStorage1d)(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_)
{
  THSYCLTensor_(setStorage)(state, self, storage_, storageOffset_,
                         {size0_}, {stride0_});
}

void THSYCLTensor_(setStorage2d)(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_)
{
  THSYCLTensor_(setStorage)(state, self, storage_, storageOffset_,
                         {size0_, size1_},
                         {stride0_, stride1_});
}

void THSYCLTensor_(setStorage3d)(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_)
{
  THSYCLTensor_(setStorage)(state, self, storage_, storageOffset_,
                         {size0_, size1_, size2_},
                         {stride0_, stride1_, stride2_});
}

void THSYCLTensor_(setStorage4d)(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_,
                             int64_t size3_, int64_t stride3_)
{

  int64_t size[4] = {size0_, size1_, size2_, size3_};
  int64_t stride[4] = {stride0_, stride1_, stride2_, stride3_};

  THSYCLTensor_(setStorage)(state, self, storage_, storageOffset_, size, stride);
}

void THSYCLTensor_(narrow)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension, int64_t firstIndex, int64_t size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->dim()), 3, "out of range");
  THArgCheck( firstIndex >= 0, 4, "out of range");
  THArgCheck( size >= 0, 5, "out of range");
  THArgCheck(firstIndex+size <= src->size(dimension), 5, "out of range");

  THSYCLTensor_(set)(state, self, src);

  if (firstIndex > 0) {
    self->set_storage_offset(self->storage_offset() + firstIndex*self->stride(dimension));
  }

  self->set_size(dimension, size);
}

void THSYCLTensor_(select)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension, int64_t sliceIndex)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(src->dim() > 0, 1, "cannot select on a 0-dim tensor");
  THArgCheck((dimension >= 0) && (dimension < src->dim()), 3, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size(dimension)), 4, "out of range");

  THSYCLTensor_(set)(state, self, src);
  THSYCLTensor_(narrow)(state, self, NULL, dimension, sliceIndex, 1);
  for(d = dimension; d < self->dim()-1; d++)
  {
    self->set_size(d, self->size(d+1));
    self->set_stride(d, self->stride(d+1));
  }
  self->resize_dim((unsigned int)(self->dim() - 1));
}

void THSYCLTensor_(transpose)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension1, int dimension2)
{
  int64_t z;

  if(!src)
    src = self;

  THArgCheck( (dimension1 >= 0) && (dimension1 < THTensor_nDimensionLegacyNoScalars(src)), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < THTensor_nDimensionLegacyNoScalars(src)), 2, "out of range");

  THSYCLTensor_(set)(state, self, src);

  if(dimension1 == dimension2)
    return;

  z = self->stride(dimension1);
  self->set_stride(dimension1, self->stride(dimension2));
  self->set_stride(dimension2, z);
  z = self->size(dimension1);
  self->set_size(dimension1, self->size(dimension2));
  self->set_size(dimension2, z);
}

void THSYCLTensor_(unfold)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension, int64_t size, int64_t step)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(dimension < THTensor_nDimensionLegacyNoScalars(src), 2, "out of range");
  THArgCheck(size <= THTensor_sizeLegacyNoScalars(src, dimension), 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THSYCLTensor_(set)(state, self, src);

  std::vector<int64_t> newSize(self->dim() + 1);
  std::vector<int64_t> newStride(self->dim() + 1);

  newSize[self->dim()] = size;
  newStride[self->dim()] = THTensor_strideLegacyNoScalars(self, dimension);
  for(d = 0; d < self->dim(); d++)
  {
    auto self_size = THTensor_sizeLegacyNoScalars(self, d);
    auto self_stride = THTensor_strideLegacyNoScalars(self, d);
    if(d == dimension)
    {
      newSize[d] = (self_size - size) / step + 1;
      newStride[d] = step*self_stride;
    }
    else
    {
      newSize[d] = self_size;
      newStride[d] = self_stride;
    }
  }

  self->set_sizes_and_strides(newSize, newStride);
}

/* we have to handle the case where the result is a number */
void THSYCLTensor_(squeeze)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src)
{
  int ndim = 0;
  int d;

  if(!src)
    src = self;

  THSYCLTensor_(set)(state, self, src);

  for(d = 0; d < src->dim(); d++)
  {
    if(src->size(d) != 1)
    {
      if(d != ndim)
      {
        self->set_size(ndim, src->size(d));
        self->set_stride(ndim, src->stride(d));
      }
      ndim++;
    }
  }

  self->resize_dim(ndim);
}

void THSYCLTensor_(squeeze1d)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension)
{
  THSYCLTensor_squeeze1d(state, self, src, dimension);
}

void THSYCLTensor_(unsqueeze1d)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *src, int dimension)
{
  THSYCLTensor_unsqueeze1d(state, self, src, dimension);
}

int THSYCLTensor_(isContiguous)(THSYCLState *state, const THSYCLTensor *self)
{
  return self->is_contiguous();
}

int THSYCLTensor_(isSetTo)(THSYCLState *state, const THSYCLTensor *self, const THSYCLTensor *src)
{
  if (THTensor_getStoragePtr(self) == THTensor_getStoragePtr(src) &&
      self->storage_offset() == src->storage_offset() &&
      self->dim() == src->dim())
  {
    int d;
    for (d = 0; d < self->dim(); ++d)
    {
      if (self->size(d) != src->size(d) || self->stride(d) != src->stride(d))
        return 0;
    }
    return 1;
  }
  return 0;
}

int THSYCLTensor_(isSameSizeAs)(THSYCLState *state, const THSYCLTensor *self, const THSYCLTensor* src)
{
  int d;
  if (self->dim() != src->dim())
    return 0;
  for(d = 0; d < self->dim(); ++d)
  {
    if(self->size(d) != src->size(d))
      return 0;
  }
  return 1;
}

ptrdiff_t THSYCLTensor_(nElement)(THSYCLState *state, const THSYCLTensor *self)
{
  return THSYCLTensor_nElement(state, self);
}

void THSYCLTensor_(retain)(THSYCLState *state, THSYCLTensor *self)
{
  THSYCLTensor_retain(state, self);
}

void THSYCLTensor_(free)(THSYCLState *state, THSYCLTensor *self)
{
  THSYCLTensor_free(state, self);
}

void THSYCLTensor_(freeCopyTo)(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *dst)
{
  if(self != dst)
    THSYCLTensor_(copy)(state, dst, self);

  THSYCLTensor_(free)(state, self);
}

/*******************************************************************************/

void THSYCLTensor_(setStorageNd)(THSYCLState *state, THSYCLTensor *self, THSYCLStorage *storage, ptrdiff_t storageOffset, int nDimension, const int64_t *size, const int64_t *stride)
{
  THSYCLTensor_setStorageNd(state, self, storage, storageOffset, nDimension, size, stride);
}

void THSYCLTensor_(resizeNd)(THSYCLState *state, THSYCLTensor *self, int nDimension, const int64_t *size, const int64_t *stride)
{
  THSYCLTensor_resizeNd(state, self, nDimension, size, stride);
}

void THSYCLTensor_(set0d)(THSYCLState *state, THSYCLTensor *tensor, scalar_t value)
{
  THArgCheck(THTensor_nDimension(tensor) == 0, 1, "tensor must have no dimensions");
  THSYCLStorage_(set)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset(), value);
}


scalar_t THSYCLTensor_(get0d)(THSYCLState *state, const THSYCLTensor *tensor)
{
  THArgCheck(THTensor_nDimension(tensor) == 0, 1, "tensor must have no dimensions dimension");
  return THSYCLStorage_(get)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset());
}

void THSYCLTensor_(set1d)(THSYCLState *state, THSYCLTensor *tensor, int64_t x0, scalar_t value)
{
  THArgCheck(THTensor_nDimensionLegacyNoScalars(tensor) == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < THTensor_sizeLegacyNoScalars(tensor, 0)), 2, "out of range");
  THSYCLStorage_(set)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*THTensor_strideLegacyNoScalars(tensor, 0), value);
}

scalar_t THSYCLTensor_(get1d)(THSYCLState *state, const THSYCLTensor *tensor, int64_t x0)
{
  THArgCheck(THTensor_nDimensionLegacyNoScalars(tensor) == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < THTensor_sizeLegacyNoScalars(tensor, 0)), 2, "out of range");
  return THSYCLStorage_(get)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*THTensor_strideLegacyNoScalars(tensor, 0));
}

void THSYCLTensor_(set2d)(THSYCLState *state, THSYCLTensor *tensor, int64_t x0, int64_t x1, scalar_t value)
{
  THArgCheck(tensor->dim() == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)), 2, "out of range");
  THSYCLStorage_(set)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1), value);
}

scalar_t THSYCLTensor_(get2d)(THSYCLState *state, const THSYCLTensor *tensor, int64_t x0, int64_t x1)
{
  THArgCheck(tensor->dim() == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)), 2, "out of range");
  return THSYCLStorage_(get)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1));
}

void THSYCLTensor_(set3d)(THSYCLState *state, THSYCLTensor *tensor, int64_t x0, int64_t x1, int64_t x2, scalar_t value)
{
  THArgCheck(tensor->dim() == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)), 2, "out of range");
  THSYCLStorage_(set)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2), value);
}

scalar_t THSYCLTensor_(get3d)(THSYCLState *state, const THSYCLTensor *tensor, int64_t x0, int64_t x1, int64_t x2)
{
  THArgCheck(tensor->dim() == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)), 2, "out of range");
  return THSYCLStorage_(get)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2));
}

void THSYCLTensor_(set4d)(THSYCLState *state, THSYCLTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, scalar_t value)
{
  THArgCheck(tensor->dim() == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)) && (x3 >= 0) && (x3 < tensor->size(3)), 2, "out of range");
  THSYCLStorage_(set)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2)+x3*tensor->stride(3), value);
}

scalar_t THSYCLTensor_(get4d)(THSYCLState *state, const THSYCLTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3)
{
  THArgCheck(tensor->dim() == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size(0)) && (x1 >= 0) && (x1 < tensor->size(1)) && (x2 >= 0) && (x2 < tensor->size(2)) && (x3 >= 0) && (x3 < tensor->size(3)), 2, "out of range");
  return THSYCLStorage_(get)(state, THTensor_getStoragePtr(tensor), tensor->storage_offset()+x0*tensor->stride(0)+x1*tensor->stride(1)+x2*tensor->stride(2)+x3*tensor->stride(3));
}

int THSYCLTensor_(checkGPU)(THSYCLState *state, unsigned int nTensors, ...)
{
  c10::DeviceIndex curDev = -1;
  C10_SYCL_CHECK(c10::sycl::syclGetDevice(&curDev));
  va_list args;
  va_start(args, nTensors);
  int valid = 1;
  for (unsigned int i = 0; i < nTensors; i++) {
    THSYCLTensor* tensor = va_arg(args, THSYCLTensor*);
    if (tensor == NULL) {
      continue;
    }

    const int tensorDev = THSYCLTensor_(getDevice)(state, tensor);

    // Skips CPU tensors
    if (tensorDev == -1) { continue; }

    // Checks all tensors are on the same device
    if (tensorDev != curDev) {
      valid = 0;
      break;
    }
  }

  va_end(args);
  return valid;
}

THSYCLDescBuff THSYCLTensor_(sizeDesc)(THSYCLState *state, const THSYCLTensor *tensor) {
  const int L = THSYCL_DESC_BUFF_LEN;
  THSYCLDescBuff buf;
  char *str = buf.str;
  int n = 0;
  n += snprintf(str, L-n, "[");
  int i;
  for(i = 0; i < tensor->dim(); i++) {
    if(n >= L) break;
    n += snprintf(str+n, L-n, "%" PRId64, tensor->size(i));
    if(i < tensor->dim()-1) {
      n += snprintf(str+n, L-n, " x ");
    }
  }
  if(n < L - 2) {
    snprintf(str+n, L-n, "]");
  } else {
    snprintf(str+L-5, 5, "...]");
  }
  return buf;
}

int THSYCLTensor_(getDevice)(THSYCLState* state, const THSYCLTensor* tensor) {
    return THSYCLTensor_getDevice(state, tensor);
}


#endif



















