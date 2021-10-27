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
  return torch_ipex::cpu::kernel::vec::vec512::dil_add_softmax(a, b);
#else
  return at::softmax(at::add(a, b, 1.0f), -1);
#endif
}

} // namespace kernels
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
