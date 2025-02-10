#include <aten/Conv.h>
#include "mkl.h"
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {
namespace {
template <typename T>
std::tuple<at::Tensor, at::Tensor> causal_conv1d_update_kernel_inner(
    const at::Tensor& hidden_states,
    const at::Tensor& conv_states,
    const at::Tensor& conv_weights,
    const c10::optional<at::Tensor>& conv_bias,
    bool silu_activation) {
  auto bs = conv_states.size(0);
  auto channels = conv_states.size(1);
  auto kernel_size = conv_states.size(2);
  auto has_bias = conv_bias.has_value();
  auto bias_ptr = has_bias ? conv_bias.value().data_ptr<T>() : nullptr;
  auto conv_states_ptr = conv_states.data_ptr<T>();
  auto conv_weights_ptr = conv_weights.data_ptr<T>();
  auto hidden_states_ptr = hidden_states.data_ptr<T>();
  auto hidden_states_strideB = hidden_states.stride(0);
  auto hidden_states_strideC = hidden_states.stride(1);
  auto conv_states_strideB = conv_states.stride(0);
  auto conv_states_strideC = conv_states.stride(1);
  auto conv_states_strideK = conv_states.stride(2);
  auto conv_weights_strideC = conv_weights.stride(0);
#pragma omp parallel for collapse(2)
  for (auto bi = 0; bi < bs; bi++) {
    for (auto ci = 0; ci < channels; ci++) {
      auto conv_weights_start = ci * conv_weights_strideC;
      float out = 0.0f;
      auto conv_states_start =
          bi * conv_states_strideB + ci * conv_states_strideC;
      for (auto k = 1; k < kernel_size; k++) {
        auto conv_states_idx = conv_states_start + k * conv_states_strideK;
        out += conv_weights_ptr[conv_weights_start + k - 1] *
            conv_states_ptr[conv_states_idx];
        conv_states_ptr[conv_states_idx - conv_states_strideK] =
            conv_states_ptr[conv_states_idx];
      }
      auto hidden_states_idx =
          bi * hidden_states_strideB + ci * hidden_states_strideC;
      out += hidden_states_ptr[hidden_states_idx] *
          conv_weights_ptr[conv_weights_start + kernel_size - 1];
      conv_states_ptr
          [conv_states_start + (kernel_size - 1) * conv_states_strideK] =
              hidden_states_ptr[hidden_states_idx];
      if (has_bias) {
        out += bias_ptr[ci];
      }
      if (silu_activation) {
        out = out / (1 + expf(-out));
      }
      hidden_states_ptr[hidden_states_idx] = out;
    }
  }
  return std::make_tuple(std::move(hidden_states), std::move(conv_states));
}

std::tuple<at::Tensor, at::Tensor> causal_conv1d_update_kernel_impl(
    const at::Tensor& hidden_states,
    const at::Tensor& conv_states,
    const at::Tensor& conv_weights,
    const c10::optional<at::Tensor>& conv_bias,
    bool silu_activation) {
  if (hidden_states.scalar_type() == at::ScalarType::Float) {
    return causal_conv1d_update_kernel_inner<float>(
        hidden_states, conv_states, conv_weights, conv_bias, silu_activation);
  } else if (hidden_states.scalar_type() == at::ScalarType::BFloat16) {
    return causal_conv1d_update_kernel_inner<at::BFloat16>(
        hidden_states, conv_states, conv_weights, conv_bias, silu_activation);
  } else if (hidden_states.scalar_type() == at::ScalarType::Half) {
    return causal_conv1d_update_kernel_inner<at::Half>(
        hidden_states, conv_states, conv_weights, conv_bias, silu_activation);
  } else {
    TORCH_CHECK(
        false,
        "Only support bfloat16, float16 and float for causal_conv1d_update");
  }
}
} // anonymous namespace
IPEX_REGISTER_DISPATCH(
    causal_conv1d_update_kernel_stub,
    &causal_conv1d_update_kernel_impl);

} // namespace cpu
} // namespace torch_ipex