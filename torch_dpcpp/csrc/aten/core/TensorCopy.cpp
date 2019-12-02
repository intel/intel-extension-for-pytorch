#include <core/TensorImplUtils.h>


void TensorImpl_copy(TensorImpl* dst, TensorImpl* src) {
  if (dst == src) return;
  at::Tensor dst_wrap = TensorImpl_wrap(dst);
  at::Tensor src_wrap = TensorImpl_wrap(src);
  at::native::copy_(dst_wrap, src_wrap);
}

template <>
TensorImpl *TensorImpl_newClone<scalar_t>(TensorImpl *self) {
  TensorImpl* tensor = TensorImpl_new();
  TensorImpl_resizeAs(tensor, self);
  at::Tensor tensor_wrap = TensorImpl_wrap(tensor);
  at::Tensor self_wrap = TensorImpl_wrap(self);
  at::native::copy_(tensor_wrap, self_wrap);
  return tensor;
}

template <>
TensorImpl *TensorImpl_newContiguous<scalar_t>(TensorImpl *self)
THSYCLTensor_newContiguous
{
  if(!self->is_contiguous()) {
    return TensorImpl_newClone<scalar_t>(self);
  } else {
    TensorImpl_retain(self);
    return self;
  }
}


template <>
void TensorImpl_freeCopyTo<scalar_t>(TensorImpl *self, TensorImpl *dst) {
  if(self != dst) {
    at::Tensor dst_wrap = TensorImpl_wrap(dst);
    at::Tensor self_wrap = TensorImpl_wrap(self);
    at::native::copy_(dst_wrap, self_wrap);
  }

  TensorImpl_free(self);
}

template <>
void TensorImpl_copyIgnoringOverlaps<scalar_t>(TensorImpl* dst, TensorImpl* src) {

  AT_ERROR("not implemented TensorImpl_copyIgnoringOverlaps\n");
  // Called when we are copying into an overlapping index `dst`, but
  // we don't care which writer wins. Hacky but it works.
  // This is itself invoked by pointwiseApply2 / TensorImpl_copy in
  // case that there are write overlaps.
  // FIXME: really, overlapping writes should be illegal/an error in Torch
#if 0
 THSYCL_pointwiseApply2<scalar_t, scalar_t>(
    dst, src,
    CopyOp<scalar_t, scalar_t>(),
    ReadOnly, /* ignore overwrites */
    ReadOnly);TODO impletment it in future: jzhoulon
#endif
}

#endif

