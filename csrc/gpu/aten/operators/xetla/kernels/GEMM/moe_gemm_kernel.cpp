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

  return gpu::xetla::LaunchMoEGEMM<T, Policy>(
      queue,
      activations,
      weights,
      outputs,
      total_m,
      gemm_n,
      gemm_k,
      total_rows_for_experts,
      total_rows_for_experts_host,
      problem_count);
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

template <typename T, bool vnni_t>
cgfs_t moe_gemm_fp8(
    sycl::queue& queue,
    const T* activations,
    const uint8_t* weights,
    const fp8_format f_format,
    const float* scales,
    T* outputs,
    const int total_m,
    const int gemm_n,
    const int gemm_k,
    const int* total_rows_for_experts,
    const int* total_rows_for_experts_host,
    const int problem_count) {
  using Policy = gpu::xetla::MoEGEMMFP8Policy;

  switch (f_format) {
    case fp8_format::E4M3:
      return gpu::xetla::LaunchMoEGEMMFP8<T, fp8_format::E4M3, vnni_t, Policy>(
          queue,
          activations,
          weights,
          scales,
          outputs,
          total_m,
          gemm_n,
          gemm_k,
          total_rows_for_experts,
          total_rows_for_experts_host,
          problem_count);
    case fp8_format::E5M2:
      return gpu::xetla::LaunchMoEGEMMFP8<T, fp8_format::E5M2, vnni_t, Policy>(
          queue,
          activations,
          weights,
          scales,
          outputs,
          total_m,
          gemm_n,
          gemm_k,
          total_rows_for_experts,
          total_rows_for_experts_host,
          problem_count);
    default:
      TORCH_CHECK(
          false,
          "Error in moe_gemm_fp8: run into Unsupported fp8 format, only support FP8_E4M3 and FP8_E5M2 now. ");
  }
}

// generate the template instantiation for sycl::half and
// sycl::ext::oneapi::bfloat16
template XETLA_KERNEL_API cgfs_t moe_gemm_fp8<sycl::half, true>(
    sycl::queue& queue,
    const sycl::half* activations,
    const uint8_t* weights,
    const fp8_format f_format,
    const float* scales,
    sycl::half* outputs,
    const int total_m,
    const int gemm_n,
    const int gemm_k,
    const int* total_rows_for_experts,
    const int* total_rows_for_experts_host,
    const int problem_count);

template XETLA_KERNEL_API cgfs_t moe_gemm_fp8<sycl::half, false>(
    sycl::queue& queue,
    const sycl::half* activations,
    const uint8_t* weights,
    const fp8_format f_format,
    const float* scales,
    sycl::half* outputs,
    const int total_m,
    const int gemm_n,
    const int gemm_k,
    const int* total_rows_for_experts,
    const int* total_rows_for_experts_host,
    const int problem_count);

template XETLA_KERNEL_API cgfs_t
moe_gemm_fp8<sycl::ext::oneapi::bfloat16, true>(
    sycl::queue& queue,
    const sycl::ext::oneapi::bfloat16* activations,
    const uint8_t* weights,
    const fp8_format f_format,
    const float* scales,
    sycl::ext::oneapi::bfloat16* outputs,
    const int total_m,
    const int gemm_n,
    const int gemm_k,
    const int* total_rows_for_experts,
    const int* total_rows_for_experts_host,
    const int problem_count);

template XETLA_KERNEL_API cgfs_t
moe_gemm_fp8<sycl::ext::oneapi::bfloat16, false>(
    sycl::queue& queue,
    const sycl::ext::oneapi::bfloat16* activations,
    const uint8_t* weights,
    const fp8_format f_format,
    const float* scales,
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
