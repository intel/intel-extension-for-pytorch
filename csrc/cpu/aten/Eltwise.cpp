#include "Eltwise.h"
#include "ideep/IDeepConversions.h"

namespace torch_ipex {
namespace cpu {

at::Tensor relu_use_dst_for_bwd(
    const at::Tensor& grad_output,
    const at::Tensor& output) {
  const ideep::tensor& grady = itensor_view_from_dense(grad_output);
  const ideep::tensor& y = itensor_view_from_dense(output);
  auto grad_input = at::empty_like(output, output.options());
  ideep::tensor gradx = itensor_view_from_dense(grad_input);
  ideep::eltwise_backward::compute(
      y, grady, gradx, ideep::algorithm::eltwise_relu_use_dst_for_bwd);
  return grad_input;
}

at::Tensor sigmoid_use_dst_for_bwd(
    const at::Tensor& grad_output,
    const at::Tensor& output) {
  const ideep::tensor& grady = itensor_view_from_dense(grad_output);
  const ideep::tensor& y = itensor_view_from_dense(output);
  auto grad_input = at::empty_like(output, output.options());
  ideep::tensor gradx = itensor_view_from_dense(grad_input);
  ideep::eltwise_backward::compute(
      y, grady, gradx, ideep::algorithm::eltwise_logistic_use_dst_for_bwd);
  return grad_input;
}

} // namespace cpu
} // namespace torch_ipex
