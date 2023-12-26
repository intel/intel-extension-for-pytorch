#include "c10/core/DispatchKey.h"
#include "optimizer.h"

#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include "csrc/utils/CustomOperatorRegistration.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(packed_add_kernel_stub);

at::Tensor packed_add(
    at::Tensor& top_half_,
    at::Tensor& bot_half_,
    const at::Tensor& grad_,
    double alpha) {
  // pointer to packed_add_kernel_impl(top_half_, bot_half_, grad_, alpha);
  return packed_add_kernel_stub(kCPU, top_half_, bot_half_, grad_, alpha);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_IPEX_REGISTER_DISPATCH(
      "packed_add", torch_ipex::cpu::packed_add, at::DispatchKey::CPU);
  IPEX_OP_IPEX_REGISTER_DISPATCH(
      "packed_add", torch_ipex::cpu::packed_add, at::DispatchKey::SparseCPU);
}

} // namespace
