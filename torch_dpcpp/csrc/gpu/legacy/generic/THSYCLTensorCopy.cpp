#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "THDP/generic/THSYCLTensorCopy.cpp"
#else

void THSYCLTensor_(copy)(THSYCLState* state, THSYCLTensor* dst, THSYCLTensor* src) {
  if (dst == src) return;
  at::Tensor dst_wrap = THTensor_wrap(dst);
  at::Tensor src_wrap = THTensor_wrap(src);
  at::native::copy_(dst_wrap, src_wrap);
}

template <>
THSYCLTensor *THSYCLTensor_newClone<scalar_t>(THSYCLState *state, THSYCLTensor *self) {
  THSYCLTensor* tensor =
      THSYCLTensor_new(state, THTensor_getStoragePtr(self)->dtype());
  THSYCLTensor_resizeAs(state, tensor, self);
  at::Tensor tensor_wrap = THTensor_wrap(tensor);
  at::Tensor self_wrap = THTensor_wrap(self);
  at::native::copy_(tensor_wrap, self_wrap);
  return tensor;
}

template <>
THSYCLTensor *THSYCLTensor_newContiguous<scalar_t>(THSYCLState *state, THSYCLTensor *self)
{
  if(!self->is_contiguous()) {
    return THSYCLTensor_newClone<scalar_t>(state, self);
  } else {
    THSYCLTensor_retain(state, self);
    return self;
  }
}


template <>
void THSYCLTensor_freeCopyTo<scalar_t>(THSYCLState *state, THSYCLTensor *self, THSYCLTensor *dst) {
  if(self != dst) {
    at::Tensor dst_wrap = THTensor_wrap(dst);
    at::Tensor self_wrap = THTensor_wrap(self);
    at::native::copy_(dst_wrap, self_wrap);
  }

  THSYCLTensor_free(state, self);
}

template <>
void THSYCLTensor_copyIgnoringOverlaps<scalar_t>(THSYCLState* state, THSYCLTensor* dst, THSYCLTensor* src) {

  AT_ERROR("not implemented THSYCLTensor_copyIgnoringOverlaps\n");
  // Called when we are copying into an overlapping index `dst`, but
  // we don't care which writer wins. Hacky but it works.
  // This is itself invoked by pointwiseApply2 / THSYCLTensor_copy in
  // case that there are write overlaps.
  // FIXME: really, overlapping writes should be illegal/an error in Torch
#if 0
 THSYCL_pointwiseApply2<scalar_t, scalar_t>(
    state, dst, src,
    CopyOp<scalar_t, scalar_t>(),
    ReadOnly, /* ignore overwrites */
    ReadOnly);TODO impletment it in future: jzhoulon
#endif
}

void THSYCLTensor_(copyIgnoringOverlaps)(THSYCLState* state, THSYCLTensor* dst, THSYCLTensor* src) {
  THSYCLTensor_copyIgnoringOverlaps<scalar_t>(state, dst, src);
}

#endif

