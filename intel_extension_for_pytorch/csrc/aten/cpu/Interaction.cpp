// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "Interaction.h"
#include "csrc/autocast/autocast_mode.h"
#include "csrc/cpu/ideep/IDeepConversions.h"
#include "csrc/jit/cpu/kernels/Interaction.h"

#include <ATen/Parallel.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/function.h>
#include <algorithm>

/*
 Custom op to optimize DLRM interaction part
*/

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(interaction_forward_kernel_stub);
DEFINE_DISPATCH(interaction_backward_kernel_stub);
DEFINE_DISPATCH(dil_qinteraction_kernel_stub);

at::Tensor _interaction_forward(const std::vector<at::Tensor>& input) {
  // pointer to interaction_forward_kernel_impl(input);
  return interaction_forward_kernel_stub(kCPU, input);
}

std::vector<at::Tensor> _interaction_backward(
    const at::Tensor& grad_out,
    const std::vector<at::Tensor>& input) {
  // pointer to interaction_backward_kernel_impl(grad_out, input);
  return interaction_backward_kernel_stub(kCPU, grad_out, input);
}

at::Tensor dil_qinteraction(
    const std::vector<at::Tensor> input,
    double o_scale,
    int64_t o_zp,
    at::ScalarType o_dtype) {
  // pointer to dil_qinteraction_kernel_impl(input, o_scale, o_zp, o_dtype);
  return dil_qinteraction_kernel_stub(kCPU, input, o_scale, o_zp, o_dtype);
}

} // namespace cpu
} // namespace torch_ipex

namespace torch_ipex {

at::Tensor interaction_forward(const std::vector<at::Tensor>& input) {
  return cpu::_interaction_forward(input);
}

std::vector<at::Tensor> interaction_backward(
    const at::Tensor& grad_out,
    const std::vector<at::Tensor>& input) {
  return cpu::_interaction_backward(grad_out, input);
}

} // namespace torch_ipex

namespace {
TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
  m.def(
      torch::schema(
          "torch_ipex::interaction_forward(Tensor[] input) -> Tensor",
          c10::AliasAnalysisKind::PURE_FUNCTION),
      torch_ipex::interaction_forward);
  m.def(
      torch::schema(
          "torch_ipex::interaction_backward(Tensor grad_out, "
          "Tensor[] input) -> Tensor[]",
          c10::AliasAnalysisKind::PURE_FUNCTION),
      torch_ipex::interaction_backward);
}
} // namespace

namespace torch_ipex {
namespace autocast {

at::Tensor interaction_forward(const std::vector<at::Tensor>& input) {
  c10::impl::ExcludeDispatchKeyGuard no_autocastCPU(DispatchKey::AutocastCPU);
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_ipex::interaction_forward", "")
                       .typed<decltype(interaction_forward)>();

  auto target_type = get_autocast_dtype();

  auto type = promote_type(get_autocast_dtype(), input);
  return op.call(cpu_cached_cast(type, input));
}

TORCH_LIBRARY_IMPL(torch_ipex, AutocastCPU, m) {
  m.impl("interaction_forward", torch_ipex::autocast::interaction_forward);
}

} // namespace autocast
} // namespace torch_ipex
