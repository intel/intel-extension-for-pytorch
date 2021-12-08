#include "AddSoftmax.hpp"

#if defined(CPU_AVX512)
#include "csrc/cpu/vec512/add_softmax.h"
#endif

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace kernels {

at::Tensor DivAddSoftmax(
    at::Tensor& a,
    const at::Tensor& b,
    const float& dim_per_head) {
#if defined(CPU_AVX512)
  if (a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat) {
    return torch_ipex::cpu::kernel::vec::vec512::dil_div_add_softmax<float>(
        a, b, dim_per_head);
  } else if (
      a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16) {
    return torch_ipex::cpu::kernel::vec::vec512::dil_div_add_softmax<
        at::BFloat16>(a, b, dim_per_head);
  }
#endif
  a = at::div(a, dim_per_head);
  return at::softmax(at::add(a, b, 1.0f), -1);
}

} // namespace kernels
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
