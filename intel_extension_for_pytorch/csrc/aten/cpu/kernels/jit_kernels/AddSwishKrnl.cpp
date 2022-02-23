#include <csrc/jit/cpu/kernels/AddSwish.h>

#if defined(CPU_CAPABILITY_AVX512)
#include "csrc/cpu/vec512/add_swish.h"
#endif

namespace torch_ipex {
namespace cpu {

#if defined(DYN_DISP_BUILD)
namespace {
#endif

at::Tensor add_swish_kernel_impl(
    at::Tensor& x,
    at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c) {
#if defined(CPU_CAPABILITY_AVX512)
  if (a.scalar_type() == at::kFloat && c.scalar_type() == at::kFloat) {
    return torch_ipex::cpu::kernel::vec::vec512::dil_add_swish<float>(a, c);
  } else if (
      a.scalar_type() == at::kBFloat16 && c.scalar_type() == at::kBFloat16) {
    return torch_ipex::cpu::kernel::vec::vec512::dil_add_swish<at::BFloat16>(
        a, c);
  }
#endif
  auto lin_res = at::linear(x, b, c);
  auto sigmoid_res = at::sigmoid(lin_res);
  return at::mul(lin_res, sigmoid_res);
}

#if defined(DYN_DISP_BUILD)
} // anonymous namespace

REGISTER_DISPATCH(add_swish_kernel_stub, &add_swish_kernel_impl);

#endif

} // namespace cpu
} // namespace torch_ipex
