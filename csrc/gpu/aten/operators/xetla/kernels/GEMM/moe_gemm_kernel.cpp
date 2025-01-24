#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)

#include <vector>
#include "moe_gemm_kernel_impl.hpp"
#include "moe_gemm_policy.hpp"

namespace gpu {
namespace xetla {

template <typename T>
cgfs_t moe_gemm(
    sycl::queue& queue,
    const T* activations,
    const T* weights,
    T* outputs,
    const int total_m,
    const int gemm_n,
    const int gemm_k,
    const int* total_rows_for_experts,
    const int* total_rows_for_experts_host,
    const int problem_count) {
  using Policy = gpu::xetla::MoEGEMMPolicy;

#define MoEKernelMicro(ExpertNum)                         \
  return gpu::xetla::LaunchMoEGEMM<T, ExpertNum, Policy>( \
      queue,                                              \
      activations,                                        \
      weights,                                            \
      outputs,                                            \
      total_m,                                            \
      gemm_n,                                             \
      gemm_k,                                             \
      total_rows_for_experts,                             \
      total_rows_for_experts_host);

  switch (problem_count) {
    case 8:
      MoEKernelMicro(8);
    case 16:
      MoEKernelMicro(16);
    default:
      TORCH_CHECK(false, "unsupported expert number: ", problem_count);
      return {};
  }
#undef MoEKernelMicro
}

// generate the template instantiation for sycl::half and
// sycl::ext::oneapi::bfloat16
template XETLA_KERNEL_API cgfs_t moe_gemm<sycl::half>(
    sycl::queue& queue,
    const sycl::half* activations,
    const sycl::half* weights,
    sycl::half* outputs,
    const int total_m,
    const int gemm_n,
    const int gemm_k,
    const int* total_rows_for_experts,
    const int* total_rows_for_experts_host,
    const int problem_count);

template XETLA_KERNEL_API cgfs_t moe_gemm<sycl::ext::oneapi::bfloat16>(
    sycl::queue& queue,
    const sycl::ext::oneapi::bfloat16* activations,
    const sycl::ext::oneapi::bfloat16* weights,
    sycl::ext::oneapi::bfloat16* outputs,
    const int total_m,
    const int gemm_n,
    const int gemm_k,
    const int* total_rows_for_experts,
    const int* total_rows_for_experts_host,
    const int problem_count);

} // namespace xetla
} // namespace gpu

#endif
