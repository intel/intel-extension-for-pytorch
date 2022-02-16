#include "ConcatBnRelu.h"

#include "csrc/utils/ipex_op_profile.h"

#if defined(CPU_CAPABILITY_AVX512)
#include "csrc/cpu/vec512/concat_bn_relu.h"
#endif
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
    const at::Tensor& bn_beta,
    const c10::optional<at::Tensor>& bn_scale,
    const c10::optional<at::Tensor>& bn_bias,
    const c10::optional<at::Tensor>& bn_mean,
    const c10::optional<at::Tensor>& bn_var,
    bool bn_training,
    double bn_momentum,
    double bn_eps,
    bool bn_cudnn_enabled,
    int dim) {
  IPEX_RECORD_FUNCTION("ConcatBnRelu", std::vector<c10::IValue>({}));

#if defined(DYN_DISP_BUILD)
  return concat_bn_relu_kernel_stub(
      kCPU,
      a,
      bn_beta,
      bn_scale,
      bn_bias,
      bn_mean,
      bn_var,
      bn_training,
      bn_momentum,
      bn_eps,
      bn_cudnn_enabled,
      dim);
#else
  return concat_bn_relu_kernel_impl(
      a,
      bn_beta,
      bn_scale,
      bn_bias,
      bn_mean,
      bn_var,
      bn_training,
      bn_momentum,
      bn_eps,
      bn_cudnn_enabled,
      dim);
#endif
}

} // namespace cpu
} // namespace torch_ipex