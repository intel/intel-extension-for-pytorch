#include "Punica.h"
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include "csrc/utils/CustomOperatorRegistration.h"

namespace torch_ipex {
namespace cpu {

IPEX_DEFINE_DISPATCH(punica_bgmv_shrink_kernel_stub);
IPEX_DEFINE_DISPATCH(punica_sgmv_shrink_kernel_stub);
IPEX_DEFINE_DISPATCH(punica_bgmv_expand_kernel_stub);
IPEX_DEFINE_DISPATCH(punica_sgmv_expand_kernel_stub);
IPEX_DEFINE_DISPATCH(punica_bgmv_expand_slice_kernel_stub);
IPEX_DEFINE_DISPATCH(punica_sgmv_expand_slice_kernel_stub);

at::Tensor punica_bgmv_shrink_forward_cpu(
    at::Tensor& out,
    at::Tensor& input,
    at::Tensor& weights,
    at::Tensor& indicies,
    const double scale) {
  punica_bgmv_shrink_kernel_stub(kCPU, out, input, weights, indicies, scale);
  return out;
}

at::Tensor punica_sgmv_shrink_forward_cpu(
    at::Tensor& out,
    at::Tensor& input,
    at::Tensor& weights,
    at::Tensor& indicies,
    at::Tensor& seq_lens,
    const double scale) {
  punica_sgmv_shrink_kernel_stub(
      kCPU, out, input, weights, indicies, seq_lens, scale);
  return out;
}

at::Tensor punica_bgmv_expand_forward_cpu(
    at::Tensor& out,
    at::Tensor& input,
    at::Tensor& weights,
    at::Tensor& indicies,
    bool add_inputs) {
  punica_bgmv_expand_kernel_stub(
      kCPU, out, input, weights, indicies, add_inputs);
  return out;
}

at::Tensor punica_sgmv_expand_forward_cpu(
    at::Tensor& out,
    at::Tensor& input,
    at::Tensor& weights,
    at::Tensor& indicies,
    at::Tensor& seq_lens,
    bool add_inputs) {
  punica_sgmv_expand_kernel_stub(
      kCPU, out, input, weights, indicies, seq_lens, add_inputs);
  return out;
}

at::Tensor punica_bgmv_expand_slice_forward_cpu(
    at::Tensor& out,
    at::Tensor& input,
    at::Tensor& weights,
    at::Tensor& indicies,
    int64_t slice_offset,
    int64_t slice_size,
    bool add_inputs) {
  punica_bgmv_expand_slice_kernel_stub(
      kCPU,
      out,
      input,
      weights,
      indicies,
      slice_offset,
      slice_size,
      add_inputs);
  return out;
}

at::Tensor punica_sgmv_expand_slice_forward_cpu(
    at::Tensor& out,
    at::Tensor& input,
    at::Tensor& weights,
    at::Tensor& indicies,
    at::Tensor& seq_lens,
    int64_t slice_offset,
    int64_t slice_size,
    bool add_inputs) {
  punica_sgmv_expand_slice_kernel_stub(
      kCPU,
      out,
      input,
      weights,
      indicies,
      seq_lens,
      slice_offset,
      slice_size,
      add_inputs);
  return out;
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
      "punica_sgmv_shrink",
      torch_ipex::cpu::punica_sgmv_shrink_forward_cpu,
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
      "punica_sgmv_expand",
      torch_ipex::cpu::punica_sgmv_expand_forward_cpu,
      c10::DispatchKey::CPU);
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  IPEX_OP_REGISTER_DISPATCH(
      "punica_bgmv_expand_slice",
      torch_ipex::cpu::punica_bgmv_expand_slice_forward_cpu,
      c10::DispatchKey::CPU);
}

TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  IPEX_OP_REGISTER_DISPATCH(
      "punica_sgmv_expand_slice",
      torch_ipex::cpu::punica_sgmv_expand_slice_forward_cpu,
      c10::DispatchKey::CPU);
}

} // namespace
