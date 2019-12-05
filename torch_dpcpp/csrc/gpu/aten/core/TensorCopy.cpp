#include <ATen/native/TensorIterator.h>
#include <ATen/core/Tensor.h>

#include <core/TensorImplUtils.h>
#include <core/TensorCopy.h>
#include <functions/Copy.h>

namespace at {
namespace native {

#define BUILD_TENSOR_ITER(dst, src, iter) \
  auto iter = TensorIterator();           \
  iter.add_output(dst);                   \
  iter.add_input(src);                    \
  iter.dont_resize_outputs();             \
  iter.dont_compute_common_dtype();       \
  iter.build();

void TensorImpl_copy(TensorImpl* dst, TensorImpl* src) {
  if (dst == src) return;
  at::Tensor dst_wrap = TensorImpl_wrap(dst);
  at::Tensor src_wrap = TensorImpl_wrap(src);
  BUILD_TENSOR_ITER(dst_wrap, src_wrap, iter)
  at::native::copy_kernel_sycl(iter);
}

template <typename scalar_t>
TensorImpl *TensorImpl_newClone(TensorImpl *self) {
  TensorImpl* tensor = TensorImpl_new();
  TensorImpl_resizeAs(tensor, self);
  at::Tensor tensor_wrap = TensorImpl_wrap(tensor);
  at::Tensor self_wrap = TensorImpl_wrap(self);
  BUILD_TENSOR_ITER(tensor_wrap, self_wrap, iter)
  at::native::copy_kernel_sycl(iter);
  return tensor;
}

template <typename scalar_t>
TensorImpl *TensorImpl_newContiguous(TensorImpl *self)
{
  if(!self->is_contiguous()) {
    return TensorImpl_newClone<scalar_t>(self);
  } else {
    TensorImpl_retain(self);
    return self;
  }
}


template <typename scalar_t>
void TensorImpl_freeCopyTo(TensorImpl *self, TensorImpl *dst) {
  if(self != dst) {
    at::Tensor dst_wrap = TensorImpl_wrap(dst);
    at::Tensor self_wrap = TensorImpl_wrap(self);
    BUILD_TENSOR_ITER(dst_wrap, self_wrap, iter)
    at::native::copy_kernel_sycl(iter);
  }

  TensorImpl_free(self);
}

template <typename scalar_t>
void TensorImpl_copyIgnoringOverlaps(TensorImpl* dst, TensorImpl* src) {

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

} // namespace native
} // namespace at
