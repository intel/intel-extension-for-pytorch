// this file is main from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/layer_norm_kernel.cpp
//  and
//  https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/layer_norm.cpp

#include <csrc/aten/cpu/AddLayerNorm.h>
#include "csrc/utils/ipex_op_profile.h"

#include <torch/csrc/autograd/function.h>
#include "csrc/cpu/vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

#if defined(CPU_CAPABILITY_AVX512)
template <typename T, typename T1>
void AddLayerNormKernelImpl(
    const at::Tensor& a,
    const at::Tensor& b,
    int alpha,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    int64_t M,
    int64_t N,
    T eps,
    at::Tensor& Y) {
  DCHECK_EQ(a.numel(), M * N);
  DCHECK(!gamma.defined() || gamma.numel() == N);
  DCHECK(!beta.defined() || beta.numel() == N);
  const T* a_data = a.data_ptr<T>();
  const T* b_data = b.data_ptr<T>();
  const T1* gamma_data = gamma.defined() ? gamma.data_ptr<T1>() : nullptr;
  const T1* beta_data = beta.defined() ? beta.data_ptr<T1>() : nullptr;
  T* Y_data = Y.data_ptr<T>();
  const float c = float(1) / static_cast<float>(N);
  const bool gamma_null = gamma_data == nullptr;
  const bool beta_null = beta_data == nullptr;
  at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      at::Tensor tmp_out = at::empty({N});
      float* tmp_out_ptr = tmp_out.data_ptr<float>();
      const T* a_ptr = a_data + i * N;
      const T* b_ptr = b_data + i * N;
      T* Y_ptr = Y_data + i * N;
      float mean_val;
      float rstd_val;
      std::tie(mean_val, rstd_val) =
          kernel::_add_and_compute_mean_var<T>(a_ptr, b_ptr, N, tmp_out_ptr);
      rstd_val = std::max(rstd_val * c - mean_val * mean_val, float(0));
      rstd_val = float(1.0) / std::sqrt(rstd_val + eps);
      float scale = rstd_val;
      float bias = -rstd_val * mean_val;
      kernel::_normalize_kernel<T, T1>(
          Y_ptr, tmp_out_ptr, N, scale, bias, gamma_data, beta_data);
    }
  });
}
#endif

at::Tensor add_layer_norm_kernel_impl(
    const at::Tensor& a,
    const at::Tensor& b,
    int alpha,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    float eps) {
#if defined(CPU_CAPABILITY_AVX512)
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(a, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = a.contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  at::Tensor Y = at::native::empty_like(
      X,
      c10::nullopt /* dtype */,
      c10::nullopt /* layout */,
      c10::nullopt /* device */,
      c10::nullopt /* pin_memory */,
      at::MemoryFormat::Contiguous);
  if (a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat) {
    AddLayerNormKernelImpl<float, float>(
        X, b, alpha, weight, bias, M, N, eps, Y);
  } else if (
      a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16) {
    if (weight.defined() && weight.scalar_type() == at::kBFloat16) {
          AddLayerNormKernelImpl<at::BFloat16, at::BFloat16>(
              X, b, alpha, weight, bias, M, N, eps, Y);
    } else {
          AddLayerNormKernelImpl<at::BFloat16, float>(
              X, b, alpha, weight, bias, M, N, eps, Y);
    }
  }
  return Y;
#else
  return at::layer_norm(
      at::add(a, b, alpha), normalized_shape, weight_opt, bias_opt, eps);
#endif
}

} // anonymous namespace

REGISTER_DISPATCH(add_layer_norm_kernel_stub, &add_layer_norm_kernel_impl);

} // namespace cpu
} // namespace torch_ipex