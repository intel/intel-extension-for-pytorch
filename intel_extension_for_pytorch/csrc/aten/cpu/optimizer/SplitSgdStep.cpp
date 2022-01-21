#include "optimizer.h"

#include <torch/csrc/autograd/function.h>
#include <torch/extension.h>

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(packed_add_kernel_stub);

void packed_add(
    at::Tensor& top_half_,
    at::Tensor& bot_half_,
    const at::Tensor& grad_,
    double alpha) {
#if defined(DYN_DISP_BUILD)
  packed_add_kernel_stub(kCPU, top_half_, bot_half_, grad_, alpha);
#else
  packed_add_kernel_impl(top_half_, bot_half_, grad_, alpha);
#endif
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      "packed_add(Tensor top_half, Tensor bot_half, Tensor grad, float "
      "alpha) -> ()",
      torch_ipex::cpu::packed_add);
}

} // namespace
