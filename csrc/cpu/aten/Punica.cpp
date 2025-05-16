#include "Punica.h"
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include "csrc/utils/CustomOperatorRegistration.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(punica_bgmv_shrink_kernel_stub);
IPEX_DEFINE_DISPATCH(punica_bgmv_expand_kernel_stub);
IPEX_DEFINE_DISPATCH(punica_bgmv_expand_slice_kernel_stub);

void punica_bgmv_shrink_forward_cpu(
    at::Tensor& out,
    at::Tensor& input,
    at::Tensor& weights,
    at::Tensor& indicies,
    const double scale) {
  return punica_bgmv_shrink_kernel_stub(
      kCPU, out, input, weights, indicies, scale);
}

void punica_bgmv_expand_forward_cpu(
    at::Tensor& out,
    at::Tensor& input,
    at::Tensor& weights,
    at::Tensor& indicies,
    bool add_inputs) {
  return punica_bgmv_expand_kernel_stub(
      kCPU, out, input, weights, indicies, add_inputs);
}

void punica_bgmv_expand_slice_forward_cpu(
    at::Tensor& out,
    at::Tensor& input,
    at::Tensor& weights,
    at::Tensor& indicies,
    int64_t slice_offset,
    int64_t slice_size,
    bool add_inputs) {
  return punica_bgmv_expand_slice_kernel_stub(
      kCPU,
      out,
      input,
      weights,
      indicies,
      slice_offset,
      slice_size,
      add_inputs);
}

} // namespace cpu
} // namespace torch_ipex

namespace {

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  IPEX_OP_REGISTER_DISPATCH(
      "punica_bgmv_shrink",
      torch_ipex::cpu::punica_bgmv_shrink_forward_cpu,
      c10::DispatchKey::CPU);
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  IPEX_OP_REGISTER_DISPATCH(
      "punica_bgmv_expand",
      torch_ipex::cpu::punica_bgmv_expand_forward_cpu,
      c10::DispatchKey::CPU);
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  IPEX_OP_REGISTER_DISPATCH(
      "punica_bgmv_expand_slice",
      torch_ipex::cpu::punica_bgmv_expand_slice_forward_cpu,
      c10::DispatchKey::CPU);
}

} // namespace
