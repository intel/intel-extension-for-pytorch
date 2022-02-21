#include "AddSoftmax.h"

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(div_add_softmax_kernel_stub);

at::Tensor DivAddSoftmax(
    at::Tensor& a,
    const at::Tensor& b,
    const float& dim_per_head) {
#if defined(DYN_DISP_BUILD)
  return div_add_softmax_kernel_stub(kCPU, a, b, dim_per_head);
#else
  return div_add_softmax_kernel_impl(a, b, dim_per_head);
#endif
}

} // namespace cpu
} // namespace torch_ipex
