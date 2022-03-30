#include <csrc/aten/cpu/ConcatBnRelu.h>

#include "csrc/utils/ipex_op_profile.h"

#if defined(CPU_CAPABILITY_AVX512)
#include "csrc/cpu/vec512/concat_bn_relu.h"
#endif
#include <torch/csrc/autograd/function.h>

namespace torch_ipex {
namespace cpu {

namespace {

at::Tensor concat_bn_relu_kernel_impl(
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
  int64_t list_length = a.size();
  std::vector<int64_t> output_dim = a[0].sizes().vec();
  int64_t tensor_length = a[0].ndimension();

  // Check if the memory format is channelslast(3d) and if the channel size can
  // be divided by 16
  auto check_format_channelsize = [](at::Tensor tensor) {
    return (
        (tensor.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
         tensor.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d) &&
        tensor.size(1) % 16 == 0);
  };

  // Check the first tensor
  bool tensor_check = check_format_channelsize(a[0]);
  // Check the rest input tensors
  for (int64_t i = 1; i < list_length; ++i) {
    tensor_check = (tensor_check && check_format_channelsize(a[i]));
    for (int64_t j = 0; j < tensor_length; ++j) {
      if (j == 1) {
        output_dim[1] += a[i].size(j);
      } else {
        tensor_check = (tensor_check && a[i].size(j) == a[0].size(j));
      }
    }
  }
#if defined(CPU_CAPABILITY_AVX512)
  if (tensor_check) {
    at::Tensor output;
    if (a[0].scalar_type() == at::kBFloat16) {
      output = at::empty(
          output_dim,
          a[0].options()
              .dtype(at::kBFloat16)
              .memory_format(a[0].suggest_memory_format()));
      torch_ipex::cpu::kernel::vec::vec512::ConcatBnReluKernelImpl_ChannelsLast<
          at::BFloat16>(a, bn_scale, bn_beta, output);
    } else {
      output = at::empty(
          output_dim,
          a[0].options()
              .dtype(at::kFloat)
              .memory_format(a[0].suggest_memory_format()));
      torch_ipex::cpu::kernel::vec::vec512::ConcatBnReluKernelImpl_ChannelsLast<
          float>(a, bn_scale, bn_beta, output);
    }
    return output;
  }
#endif
  std::vector<at::Tensor> concat_input(list_length);
  for (int64_t i = 0; i < list_length; ++i)
    concat_input[i] = a[i];
  auto bn_res = at::batch_norm(
      at::cat(concat_input, (int64_t)dim),
      bn_weight,
      bn_bias,
      bn_mean,
      bn_var,
      bn_training,
      bn_momentum,
      bn_eps,
      bn_cudnn_enabled);
  return at::relu(bn_res);
}

} // anonymous namespace

REGISTER_DISPATCH(concat_bn_relu_kernel_stub, &concat_bn_relu_kernel_impl);

} // namespace cpu
} // namespace torch_ipex