#include <aten/RMSNorm.h>

#include <torch/csrc/autograd/function.h>
#include "vec/vec.h"

namespace torch_ipex {
namespace cpu {

namespace {

#if defined(CPU_CAPABILITY_AVX512)
template <typename T, typename T1>
void RMSNormKernelImpl(
    const at::Tensor& a,
    const at::Tensor& gamma,
    int64_t M,
    int64_t N,
    float eps,
    at::Tensor& Y) {
  DCHECK(a.numel() == M * N);
  DCHECK(!gamma.defined() || gamma.numel() == M * N);
  const T* a_data = a.data_ptr<T>();
  const T1* gamma_data = gamma.defined() ? gamma.data_ptr<T1>() : nullptr;
  T* Y_data = Y.data_ptr<T>();
  at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      const T* a_ptr = a_data + i * N;
      T* Y_ptr = Y_data + i * N;
      kernel::_compute_rmsnorm<T, T1>(a_ptr, N, eps, gamma_data, Y_ptr);
    }
  });
}
#endif

at::Tensor rmsnorm_kernel_impl(
    const at::Tensor& input,
    const at::Tensor& b,
    float eps) {
#if defined(CPU_CAPABILITY_AVX512)
  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();
  const int axis = input_ndim - 1;
  const int64_t M =
      c10::multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
  const int64_t N =
      c10::multiply_integers(input_shape.cbegin() + axis, input_shape.cend());
  auto X = input.contiguous();
  at::Tensor Y = at::native::empty_like(
      X,
      c10::nullopt /* dtype */,
      c10::nullopt /* layout */,
      c10::nullopt /* device */,
      c10::nullopt /* pin_memory */,
      at::MemoryFormat::Contiguous);
  if (input.scalar_type() == at::ScalarType::Float) {
    RMSNormKernelImpl<float, float>(X, b, M, N, eps, Y);
  } else if (
      input.scalar_type() == at::ScalarType::BFloat16 &&
      b.scalar_type() == at::ScalarType::Float) {
    RMSNormKernelImpl<at::BFloat16, float>(X, b, M, N, eps, Y);
  } else if (
      input.scalar_type() == at::ScalarType::BFloat16 &&
      b.scalar_type() == at::ScalarType::BFloat16) {
    RMSNormKernelImpl<at::BFloat16, at::BFloat16>(X, b, M, N, eps, Y);
  } else {
    TORCH_CHECK(false, "Unsupported input type");
  }
  return Y;
#else
  auto input1 = input.to(at::kFloat);
  auto variance = at::mean(at::pow(input1, 2), -1, true);
  auto hidden_states = at::rsqrt(at::add(variance, eps));
  return at::mul(b, at::mul(input1, hidden_states)).to(input.scalar_type());
#endif
}

} // namespace

IPEX_REGISTER_DISPATCH(rmsnorm_kernel_stub, &rmsnorm_kernel_impl);
} // namespace cpu
} // namespace torch_ipex
