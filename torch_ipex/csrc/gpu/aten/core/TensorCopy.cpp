#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>

#include <core/TensorCopy.h>
#include <core/TensorImplUtils.h>

#include <ATen/aten_ipex_type_dpcpp.h>

namespace at {
namespace dpcpp {

#define BUILD_TENSOR_ITER(dst, src, iter) \
  auto iter = TensorIterator();           \
  iter.add_output(dst);                   \
  iter.add_input(src);                    \
  iter.dont_resize_outputs();             \
  iter.dont_compute_common_dtype();       \
  iter.build();

void TensorImpl_copy(TensorImpl* dst, TensorImpl* src) {
  if (dst == src)
    return;
  auto dst_ = TensorImpl_wrap(dst);
  auto src_ = TensorImpl_wrap(src);
  at::AtenIpexTypeDPCPP::copy_(dst_, src_, false);
}

template <typename scalar_t>
TensorImpl* TensorImpl_newClone(TensorImpl* self) {
  TensorImpl* tensor = TensorImpl_new(self->is_quantized());
  TensorImpl_resizeAs(tensor, self);
  auto dst_ = TensorImpl_wrap(tensor);
  auto src_ = TensorImpl_wrap(self);
  at::AtenIpexTypeDPCPP::copy_(dst_, src_, false);
  return tensor;
}

template <typename scalar_t>
TensorImpl* TensorImpl_newContiguous(TensorImpl* self) {
  if (!self->is_contiguous()) {
    return TensorImpl_newClone<scalar_t>(self);
  } else {
    TensorImpl_retain(self);
    return self;
  }
}

template <typename scalar_t>
void TensorImpl_freeCopyTo(TensorImpl* self, TensorImpl* dst) {
  if (self != dst) {
    auto dst_ = TensorImpl_wrap(dst);
    auto src_ = TensorImpl_wrap(self);
    at::AtenIpexTypeDPCPP::copy_(dst_, src_, false);
  }

  TensorImpl_free(self);
}

template <typename scalar_t>
void TensorImpl_copyIgnoringOverlaps(TensorImpl* dst, TensorImpl* src) {
  TORCH_CHECK(0, "not implemented TensorImpl_copyIgnoringOverlaps\n");
#if 0
 THDPCPP_pointwiseApply2<scalar_t, scalar_t>(
    dst, src,
    CopyOp<scalar_t, scalar_t>(),
    ReadOnly, /* ignore overwrites */
    ReadOnly);TODO impletment it in future: jzhoulon
#endif
}

} // namespace dpcpp
} // namespace at
