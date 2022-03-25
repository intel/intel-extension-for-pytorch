#include <csrc/aten/cpu/DivSoftmax.h>

#if defined(CPU_CAPABILITY_AVX512)
#include "csrc/cpu/vec512/add_softmax.h"
#endif

namespace torch_ipex {
namespace cpu {

namespace {

at::Tensor div_maskedfill_softmax_kernel_impl(
    at::Tensor& a,
    const at::Tensor& b,
    const at::IntArrayRef& mask_shape,
    const float& fill,
    const float& dim_per_head) {
#if defined(CPU_CAPABILITY_AVX512)
  if (a.scalar_type() == at::kFloat) {
    return torch_ipex::cpu::kernel::vec::vec512::dil_div_maskfill_softmax<
        float>(a, b, fill, dim_per_head);
  } else if (a.scalar_type() == at::kBFloat16) {
    return torch_ipex::cpu::kernel::vec::vec512::dil_div_maskfill_softmax<
        at::BFloat16>(a, b, fill, dim_per_head);
  }
#endif
  // convert the mask back to bool for fallback path
  auto _b = b.toType(at::kBool);
  a = at::div(a, dim_per_head);
  auto expand_mask = _b.view(mask_shape).expand_as(a);
  auto a_fill = a.masked_fill_(expand_mask, fill);
  return at::softmax(a_fill, -1);
}

} // anonymous namespace

REGISTER_DISPATCH(
    div_maskedfill_softmax_kernel_stub,
    &div_maskedfill_softmax_kernel_impl);

} // namespace cpu
} // namespace torch_ipex
