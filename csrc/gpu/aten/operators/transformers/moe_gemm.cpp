#include "../xetla/moe_gemm.h"
#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/record_function.h>
#include <runtime/Device.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "../comm/ATDispatch.h"
#include "utils/CustomOperatorRegistration.h"

namespace at {
namespace AtenIpexTypeXPU {

/**
 * @brief Performs matrix multiplication used in the MOE (Mixture of Experts)
 * algorithm.
 *
 * This function takes two input matrices, `matrix_a` and `matrix_b`, and
 * performs matrix multiplication used for MOE. The resulting matrix is stored
 * in the `output` tensor.
 *
 * @param matrix_a The input matrix A with shape [total_m, gemm_k].
 * @param matrix_b The input matrix B with shape [n_experts, gemm_k, gemm_n].
 * @param rows_for_experts The tensor containing the number of rows(tokens) for
 * each expert.
 * @param rows_for_experts_host The tensor containing the number of rows for
 * each expert (host version), used for CPU dispatch
 * @param n_experts The number of experts.
 * @return The output tensor with shape [total_m, gemm_n].
 *
 * @throws std::runtime_error If the input tensors have invalid shapes or
 * unsupported data types.
 */
Tensor fused_moe_gemm(
    const Tensor& matrix_a, // [total_m, gemm_k]
    const Tensor& matrix_b, // [n_experts, gemm_k, gemm_n]
    const Tensor& rows_for_experts, //[n_experts]
    const Tensor& rows_for_experts_host, //[n_experts]
    const int64_t n_experts) {
  RECORD_FUNCTION("xetla_fused_moe_gemm", {});

  int total_m = matrix_a.sizes()[0];
  int gemm_k = matrix_a.sizes()[1];
  auto matrix_b_shape = matrix_b.sizes().vec();
  int gemm_n = matrix_b_shape[2];

  TORCH_CHECK(matrix_b_shape.size() == 3, "matrix_b must be 3D");
  TORCH_CHECK(
      matrix_b_shape[0] == n_experts,
      "matrix_b must have n_experts as the first dimension");
  TORCH_CHECK(
      matrix_b_shape[1] == gemm_k,
      "matrix_b must have the same size as matrix_a in the second dimension");
  TORCH_CHECK(
      matrix_b_shape[0] == rows_for_experts.size(0),
      "rows_for_experts must have the same size as the first dimension of matrix_b");

  auto output = at::empty({total_m, gemm_n}, matrix_a.options());
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)

  auto queue = dpcppGetCurrentQueue();

  if (matrix_a.scalar_type() == at::kHalf) {
    auto cgfs = gpu::xetla::moe_gemm(
        queue,
        reinterpret_cast<sycl::half*>(matrix_a.data_ptr()),
        reinterpret_cast<sycl::half*>(matrix_b.data_ptr()),
        reinterpret_cast<sycl::half*>(output.data_ptr()),
        total_m,
        gemm_n,
        gemm_k,
        reinterpret_cast<int*>(rows_for_experts.data_ptr()),
        reinterpret_cast<int*>(rows_for_experts_host.data_ptr()),
        n_experts);
    DPCPP_Q_SUBMIT_CGFS(queue, cgfs);
  } else if (matrix_a.scalar_type() == at::kBFloat16) {
    auto cgfs = gpu::xetla::moe_gemm<sycl::ext::oneapi::bfloat16>(
        queue,
        reinterpret_cast<sycl::ext::oneapi::bfloat16*>(matrix_a.data_ptr()),
        reinterpret_cast<sycl::ext::oneapi::bfloat16*>(matrix_b.data_ptr()),
        reinterpret_cast<sycl::ext::oneapi::bfloat16*>(output.data_ptr()),
        total_m,
        gemm_n,
        gemm_k,
        reinterpret_cast<int*>(rows_for_experts.data_ptr()),
        reinterpret_cast<int*>(rows_for_experts_host.data_ptr()),
        n_experts);
    DPCPP_Q_SUBMIT_CGFS(queue, cgfs);
  } else {
    TORCH_CHECK(
        false,
        "Error in moegemm: run into Unsupported data type, only support half and bfloat16 now. ");
  }
#else
  AT_ERROR(
      "Fused MoE GEMM failed to compile because the current hardware does not support 2D load and DPAS instructions.");
#endif
  return output;
}
} // namespace AtenIpexTypeXPU
} // namespace at

namespace {
IPEX_LIBRARY_FRAGMENT() {
  IPEX_OP_REGISTER_DISPATCH(
      "fused_moe_gemm.moe",
      at::AtenIpexTypeXPU::fused_moe_gemm,
      c10::DispatchKey::XPU);
}

} // namespace