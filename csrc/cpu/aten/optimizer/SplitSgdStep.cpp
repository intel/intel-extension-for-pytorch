#include "optimizer.h"

#include <torch/all.h>
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(packed_add_kernel_stub);

void packed_add(
    at::Tensor& top_half_,
    at::Tensor& bot_half_,
    const at::Tensor& grad_,
    double alpha) {
  // pointer to packed_add_kernel_impl(top_half_, bot_half_, grad_, alpha);
  packed_add_kernel_stub(kCPU, top_half_, bot_half_, grad_, alpha);
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
