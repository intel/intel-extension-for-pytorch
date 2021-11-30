#include <ATen/Tensor.h>
#include "ideep/ideep.hpp"

namespace torch_ipex {
namespace cpu {

at::Tensor relu_use_dst_for_bwd(
    const at::Tensor& grad_output,
    const at::Tensor& output);
at::Tensor sigmoid_use_dst_for_bwd(
    const at::Tensor& grad_output,
    const at::Tensor& output);

} // namespace cpu
} // namespace torch_ipex
