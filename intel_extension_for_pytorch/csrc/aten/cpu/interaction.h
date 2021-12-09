#include <ATen/Tensor.h>
#include <torch/extension.h>

namespace torch_ipex {

at::Tensor interaction_forward(const std::vector<at::Tensor>& input);
std::vector<at::Tensor> interaction_backward(
    const at::Tensor& grad_out,
    const std::vector<at::Tensor>& input);

} // namespace torch_ipex
