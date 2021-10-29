#include "AddSoftmax.hpp"

#if defined(CPU_AVX512)
#include "cpu/kernel/vec/vec512/add_softmax.h"
#endif

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace kernels {

at::Tensor AddSoftmax(const at::Tensor& a, const at::Tensor& b) {
#if defined(CPU_AVX512)
  if (a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat) {
    return torch_ipex::cpu::kernel::vec::vec512::dil_add_softmax<float>(a, b);
  } else if (
      a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16) {
    return torch_ipex::cpu::kernel::vec::vec512::dil_add_softmax<at::BFloat16>(
        a, b);
  }
  return at::softmax(at::add(a, b, 1.0f), -1);
#else
  return at::softmax(at::add(a, b, 1.0f), -1);
#endif
}

} // namespace kernels
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
