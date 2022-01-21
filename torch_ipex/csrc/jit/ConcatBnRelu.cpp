#include "ConcatBnRelu.h"

#if defined(CPU_AVX512)
#include "csrc/cpu/kernel/vec/vec512/concat_bn_relu.h"
#endif
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {
using Tensor = at::Tensor;

/**
 * This kernel fuses Concat+BN+ReLU as a signel operator.
 * The conditions are set as all the input tensors
 * should have the same dimension (4D or 5D), sizes,
 * ChannelsLast(3d) memory format and data type (float).
 * All the inputs should be 4D or 5D tensors. The condition
 * check should be done on graph rewrite operation before
 * calling this kernel.
 **/
Tensor ConcatBnRelu(
    const c10::List<Tensor>& a,
    const Tensor& bn_beta,
    const c10::optional<Tensor>& bn_scale,
    const c10::optional<Tensor>& bn_bias,
    const c10::optional<Tensor>& bn_mean,
    const c10::optional<Tensor>& bn_var,
    bool bn_training,
    double bn_momentum,
    double bn_eps,
    bool bn_cudnn_enabled,
    int dim) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("ConcatBnRelu", std::vector<c10::IValue>({}));
#endif
  int64_t input_len = a.size();

  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(bn_scale);
  const Tensor& bn_weight = *weight_maybe_owned;
  std::vector<long int> output_dim(a[0].ndimension());
  for (int64_t i = 0; i < a[0].ndimension(); ++i)
    output_dim[i] = a[0].size(i);
  output_dim[1] = a[0].size(1) * input_len;
  Tensor output = at::empty(
      output_dim,
      a[0].options()
          .dtype(at::kFloat)
          .memory_format(a[0].suggest_memory_format()));

#if defined(CPU_AVX512)
  torch_ipex::cpu::kernel::vec::vec512::ConcatBnReluKernelImpl_ChannelsLast<
      float>(a, bn_weight, bn_beta, output);
  return output;
#else
  std::vector<Tensor> concat_input(input_len);
  for (int64_t i = 0; i < input_len; ++i)
    concat_input[i] = a[i];
  auto bn_res = at::batch_norm(
      at::cat(concat_input, (int64_t)dim),
      bn_scale,
      bn_bias,
      bn_mean,
      bn_var,
      bn_training,
      bn_momentum,
      bn_eps,
      bn_cudnn_enabled);
  return at::relu(bn_res);
#endif
}

} // namespace cpu
} // namespace torch_ipex
