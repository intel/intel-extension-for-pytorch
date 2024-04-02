#include "DivSoftmax.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(div_maskedfill_softmax_kernel_stub);
at::Tensor DivMaskedfillSoftmax(
    at::Tensor& a,
    const at::Tensor& b,
    const at::IntArrayRef& mask_shape,
    const float& fill,
    const float& dim_per_head) {
  /*
  pointer to div_maskedfill_softmax_kernel_impl(
      a, b, mask_shape, fill, dim_per_head);
  */
  return div_maskedfill_softmax_kernel_stub(
      kCPU, a, b, mask_shape, fill, dim_per_head);
}

} // namespace cpu
} // namespace torch_ipex
