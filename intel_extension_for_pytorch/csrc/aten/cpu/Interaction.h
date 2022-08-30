#include <ATen/Tensor.h>
#include <intel_extension_for_pytorch/csrc/dyndisp/DispatchStub.h>
#include <torch/all.h>

namespace torch_ipex {

at::Tensor interaction_forward(const std::vector<at::Tensor>& input);
std::vector<at::Tensor> interaction_backward(
    const at::Tensor& grad_out,
    const std::vector<at::Tensor>& input);

} // namespace torch_ipex

namespace torch_ipex {
namespace cpu {

namespace {

at::Tensor interaction_forward_kernel_impl(
    const std::vector<at::Tensor>& input);

std::vector<at::Tensor> interaction_backward_kernel_impl(
    const at::Tensor& grad_out,
    const std::vector<at::Tensor>& input);

at::Tensor dil_qinteraction_kernel_impl(
    const std::vector<at::Tensor> input,
    double o_scale,
    int64_t o_zp,
    at::ScalarType o_dtype);

} // namespace

using interaction_forward_kernel_fn =
    at::Tensor (*)(const std::vector<at::Tensor>&);
DECLARE_DISPATCH(
    interaction_forward_kernel_fn,
    interaction_forward_kernel_stub);

using interaction_backward_kernel_fn = std::vector<at::Tensor> (*)(
    const at::Tensor&,
    const std::vector<at::Tensor>&);
DECLARE_DISPATCH(
    interaction_backward_kernel_fn,
    interaction_backward_kernel_stub);

using dil_qinteraction_kernel_fn = at::Tensor (*)(
    const std::vector<at::Tensor>,
    double,
    int64_t,
    at::ScalarType);
DECLARE_DISPATCH(dil_qinteraction_kernel_fn, dil_qinteraction_kernel_stub);

} // namespace cpu
} // namespace torch_ipex
