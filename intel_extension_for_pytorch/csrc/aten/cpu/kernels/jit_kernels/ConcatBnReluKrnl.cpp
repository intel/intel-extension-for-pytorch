#include <csrc/jit/cpu/kernels/ConcatBnRelu.h>

#include "csrc/utils/ipex_op_profile.h"

#if defined(CPU_CAPABILITY_AVX512)
#include "csrc/cpu/vec512/concat_bn_relu.h"
#endif
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

#if defined(DYN_DISP_BUILD)
namespace {
#endif

at::Tensor concat_bn_relu_kernel_impl(
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
  int64_t list_length = a.size();

  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(bn_scale);
  const at::Tensor& bn_weight = *weight_maybe_owned;
  std::vector<long int> output_dim(a[0].ndimension());
  for (int64_t i = 0; i < list_length; ++i) {
    output_dim[1] += a[i].size(1);
  }
  for (int64_t i = 0; i < a[0].ndimension(); ++i) {
    if (i != 1) {
      output_dim[i] = a[0].size(i);
    }
  }
  at::Tensor output = at::empty(
      output_dim,
      a[0].options()
          .dtype(at::kFloat)
          .memory_format(a[0].suggest_memory_format()));

#if defined(CPU_CAPABILITY_AVX512)
  torch_ipex::cpu::kernel::vec::vec512::ConcatBnReluKernelImpl_ChannelsLast<
      float>(a, bn_weight, bn_beta, output);
  return output;
#else
  std::vector<at::Tensor> concat_input(list_length);
  for (int64_t i = 0; i < list_length; ++i)
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

#if defined(DYN_DISP_BUILD)
} // anonymous namespace

REGISTER_DISPATCH(concat_bn_relu_kernel_stub, &concat_bn_relu_kernel_impl);

#endif

} // namespace cpu
} // namespace torch_ipex