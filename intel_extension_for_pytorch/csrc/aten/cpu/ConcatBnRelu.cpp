#include "ConcatBnRelu.h"

#include "csrc/utils/ipex_op_profile.h"

#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

DEFINE_DISPATCH(concat_bn_relu_kernel_stub);

/**
 * This kernel fuses Concat+BN+ReLU as a signel operator.
 * The conditions are set as all the input tensors
 * should have the same dimension (4D or 5D), sizes,
 * ChannelsLast(3d) memory format and data type (float).
 * All the inputs should be 4D or 5D tensors. The condition
 * check should be done on graph rewrite operation before
 * calling this kernel.
 **/
at::Tensor ConcatBnRelu(
    const c10::List<at::Tensor>& a,
    const at::Tensor& bn_scale,
    const at::Tensor& bn_beta,
    const c10::optional<at::Tensor>& bn_weight,
    const c10::optional<at::Tensor>& bn_bias,
    const c10::optional<at::Tensor>& bn_mean,
    const c10::optional<at::Tensor>& bn_var,
    bool bn_training,
    double bn_momentum,
    double bn_eps,
    bool bn_cudnn_enabled,
    int dim) {
  IPEX_RECORD_FUNCTION("ipex::concat_bn_relu", c10::ArrayRef<c10::IValue>({}));

  /*
  pointer to concat_bn_relu_kernel_impl(
      a,
      bn_scale,
      bn_beta,
      bn_weight,
      bn_bias,
      bn_mean,
      bn_var,
      bn_training,
      bn_momentum,
      bn_eps,
      bn_cudnn_enabled,
      dim);
  */
  return concat_bn_relu_kernel_stub(
      kCPU,
      a,
      bn_scale,
      bn_beta,
      bn_weight,
      bn_bias,
      bn_mean,
      bn_var,
      bn_training,
      bn_momentum,
      bn_eps,
      bn_cudnn_enabled,
      dim);
}

} // namespace cpu
} // namespace torch_ipex