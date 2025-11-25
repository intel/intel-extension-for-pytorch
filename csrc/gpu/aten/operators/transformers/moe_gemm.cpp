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
    const c10::optional<at::Tensor>& matrix_b_scale_inv, // [n_experts]
    const c10::optional<bool> vnni_transform, // [n_experts]
    const Tensor& rows_for_experts, //[n_experts]
    const Tensor& rows_for_experts_host, //[n_experts]
    const int64_t n_experts) {
  RECORD_FUNCTION("xetla_fused_moe_gemm", {});

  int total_m = matrix_a.sizes()[0];
  int gemm_k = matrix_a.sizes()[1];
  auto matrix_b_shape = matrix_b.sizes().vec();
  int gemm_n = matrix_b_shape[2];

  int n_experts_local = matrix_b_shape[0];
  int n_experts_aligned = (n_experts_local + 7) / 8 * 8; // align to 8

  TORCH_CHECK(matrix_b_shape.size() == 3, "matrix_b must be 3D");
  TORCH_CHECK(
      matrix_b_shape[0] == n_experts,
      "matrix_b must have n_experts as the first dimension");
  TORCH_CHECK(
      matrix_b_shape[1] == gemm_k,
      "matrix_b must have the same size as matrix_a in the second dimension");
  TORCH_CHECK(
      n_experts_aligned == rows_for_experts.size(0),
      "The size of experts must be aligned to an integer multiple of 8.");

  auto output = at::empty({total_m, gemm_n}, matrix_a.options());
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)

  auto queue = dpcppGetCurrentQueue();

  bool is_fp8 = matrix_b_scale_inv.has_value();
  if (is_fp8) {
    auto fp8_dtype = matrix_b.scalar_type();
    bool is_vnni_transform =
        vnni_transform.has_value() && vnni_transform.value();
    gpu::xetla::fp8_format f_format;
    if (fp8_dtype == at::kFloat8_e4m3fn) {
      f_format = gpu::xetla::fp8_format::E4M3;
    } else if (fp8_dtype == at::kFloat8_e5m2) {
      f_format = gpu::xetla::fp8_format::E5M2;
    } else {
      TORCH_CHECK(
          false,
          "Error in moegemm: run into Unsupported data type for matrix_b, only support float8_e4m3fn and float8_e5m2 now. ");
    }

    if (matrix_a.scalar_type() == at::kHalf && is_vnni_transform) {
      auto cgfs = gpu::xetla::moe_gemm_fp8<sycl::half, true>(
          queue,
          reinterpret_cast<sycl::half*>(matrix_a.data_ptr()),
          reinterpret_cast<uint8_t*>(matrix_b.data_ptr()),
          f_format,
          reinterpret_cast<float*>(matrix_b_scale_inv->data_ptr()),
          reinterpret_cast<sycl::half*>(output.data_ptr()),
          total_m,
          gemm_n,
          gemm_k,
          reinterpret_cast<int*>(rows_for_experts.data_ptr()),
          reinterpret_cast<int*>(rows_for_experts_host.data_ptr()),
          n_experts);
      DPCPP_Q_SUBMIT_CGFS(queue, cgfs);
    } else if (matrix_a.scalar_type() == at::kBFloat16 && is_vnni_transform) {
      auto cgfs = gpu::xetla::moe_gemm_fp8<sycl::ext::oneapi::bfloat16, true>(
          queue,
          reinterpret_cast<sycl::ext::oneapi::bfloat16*>(matrix_a.data_ptr()),
          reinterpret_cast<uint8_t*>(matrix_b.data_ptr()),
          f_format,
          reinterpret_cast<float*>(matrix_b_scale_inv->data_ptr()),
          reinterpret_cast<sycl::ext::oneapi::bfloat16*>(output.data_ptr()),
          total_m,
          gemm_n,
          gemm_k,
          reinterpret_cast<int*>(rows_for_experts.data_ptr()),
          reinterpret_cast<int*>(rows_for_experts_host.data_ptr()),
          n_experts);
      DPCPP_Q_SUBMIT_CGFS(queue, cgfs);
    } else if (matrix_a.scalar_type() == at::kHalf && (!is_vnni_transform)) {
      auto cgfs = gpu::xetla::moe_gemm_fp8<sycl::half, false>(
          queue,
          reinterpret_cast<sycl::half*>(matrix_a.data_ptr()),
          reinterpret_cast<uint8_t*>(matrix_b.data_ptr()),
          f_format,
          reinterpret_cast<float*>(matrix_b_scale_inv->data_ptr()),
          reinterpret_cast<sycl::half*>(output.data_ptr()),
          total_m,
          gemm_n,
          gemm_k,
          reinterpret_cast<int*>(rows_for_experts.data_ptr()),
          reinterpret_cast<int*>(rows_for_experts_host.data_ptr()),
          n_experts);
      DPCPP_Q_SUBMIT_CGFS(queue, cgfs);
    } else if (
        matrix_a.scalar_type() == at::kBFloat16 && (!is_vnni_transform)) {
      auto cgfs = gpu::xetla::moe_gemm_fp8<sycl::ext::oneapi::bfloat16, false>(
          queue,
          reinterpret_cast<sycl::ext::oneapi::bfloat16*>(matrix_a.data_ptr()),
          reinterpret_cast<uint8_t*>(matrix_b.data_ptr()),
          f_format,
          reinterpret_cast<float*>(matrix_b_scale_inv->data_ptr()),
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
  } else if (matrix_a.scalar_type() == at::kHalf) {
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

Tensor fused_moe_gemm_persistent(
    const Tensor& matrix_a, // [total_m, gemm_k]
    const Tensor& matrix_b, // [n_experts, gemm_k, gemm_n]
    const c10::optional<at::Tensor>& matrix_b_scale_inv, // [n_experts]
    const Tensor& rows_for_experts, //[n_experts]
    const int64_t n_experts) {
  RECORD_FUNCTION("xetla_fused_moe_gemm_persistent", {});

  int total_m = matrix_a.sizes()[0];
  int gemm_k = matrix_a.sizes()[1];
  auto matrix_b_shape = matrix_b.sizes().vec();
  int gemm_n = matrix_b_shape[2];

  int n_experts_local = matrix_b_shape[0];
  int n_experts_aligned = (n_experts_local + 7) / 8 * 8; // align to 8

  TORCH_CHECK(matrix_b_shape.size() == 3, "matrix_b must be 3D");
  TORCH_CHECK(
      matrix_b_shape[0] == n_experts,
      "matrix_b must have n_experts as the first dimension");
  TORCH_CHECK(
      matrix_b_shape[1] == gemm_k,
      "matrix_b must have the same size as matrix_a in the second dimension");
  TORCH_CHECK(
      n_experts_aligned == rows_for_experts.size(0),
      "The size of experts must be aligned to an integer multiple of 8.");

  auto output = at::empty({total_m, gemm_n}, matrix_a.options());
#if defined(USE_XETLA) && defined(USE_XETLA_XE_HPC)

  Tensor atomic_buffer =
      at::empty({static_cast<long>(1)}, matrix_a.options().dtype(at::kInt));

  auto queue = dpcppGetCurrentQueue();

  bool is_fp8 = matrix_b_scale_inv.has_value();
  if (is_fp8) {
    auto fp8_dtype = matrix_b.scalar_type();
    gpu::xetla::fp8_format f_format;
    if (fp8_dtype == at::kFloat8_e4m3fn) {
      f_format = gpu::xetla::fp8_format::E4M3;
    } else if (fp8_dtype == at::kFloat8_e5m2) {
      f_format = gpu::xetla::fp8_format::E5M2;
    } else {
      TORCH_CHECK(
          false,
          "Error in moegemm: run into Unsupported data type for matrix_b, only support float8_e4m3fn and float8_e5m2 now. ");
    }

    if (matrix_a.scalar_type() == at::kHalf) {
      auto cgfs = gpu::xetla::persistent_moe_gemm_fp8<sycl::half>(
          queue,
          reinterpret_cast<sycl::half*>(matrix_a.data_ptr()),
          reinterpret_cast<uint8_t*>(matrix_b.data_ptr()),
          f_format,
          reinterpret_cast<float*>(matrix_b_scale_inv->data_ptr()),
          reinterpret_cast<sycl::half*>(output.data_ptr()),
          total_m,
          gemm_n,
          gemm_k,
          reinterpret_cast<int*>(rows_for_experts.data_ptr()),
          static_cast<int*>(atomic_buffer.data_ptr()),
          n_experts);
      DPCPP_Q_SUBMIT_CGFS(queue, cgfs);
    } else if (matrix_a.scalar_type() == at::kBFloat16) {
      auto cgfs =
          gpu::xetla::persistent_moe_gemm_fp8<sycl::ext::oneapi::bfloat16>(
              queue,
              reinterpret_cast<sycl::ext::oneapi::bfloat16*>(
                  matrix_a.data_ptr()),
              reinterpret_cast<uint8_t*>(matrix_b.data_ptr()),
              f_format,
              reinterpret_cast<float*>(matrix_b_scale_inv->data_ptr()),
              reinterpret_cast<sycl::ext::oneapi::bfloat16*>(output.data_ptr()),
              total_m,
              gemm_n,
              gemm_k,
              reinterpret_cast<int*>(rows_for_experts.data_ptr()),
              static_cast<int*>(atomic_buffer.data_ptr()),
              n_experts);
      DPCPP_Q_SUBMIT_CGFS(queue, cgfs);
    } else {
      TORCH_CHECK(
          false,
          "Error in moegemm: run into Unsupported data type, only support half and bfloat16 now. ");
    }
  } else if (matrix_a.scalar_type() == at::kHalf) {
    auto cgfs = gpu::xetla::persistent_moe_gemm(
        queue,
        reinterpret_cast<sycl::half*>(matrix_a.data_ptr()),
        reinterpret_cast<sycl::half*>(matrix_b.data_ptr()),
        reinterpret_cast<sycl::half*>(output.data_ptr()),
        total_m,
        gemm_n,
        gemm_k,
        reinterpret_cast<int*>(rows_for_experts.data_ptr()),
        static_cast<int*>(atomic_buffer.data_ptr()),
        n_experts);
    DPCPP_Q_SUBMIT_CGFS(queue, cgfs);
  } else if (matrix_a.scalar_type() == at::kBFloat16) {
    auto cgfs = gpu::xetla::persistent_moe_gemm<sycl::ext::oneapi::bfloat16>(
        queue,
        reinterpret_cast<sycl::ext::oneapi::bfloat16*>(matrix_a.data_ptr()),
        reinterpret_cast<sycl::ext::oneapi::bfloat16*>(matrix_b.data_ptr()),
        reinterpret_cast<sycl::ext::oneapi::bfloat16*>(output.data_ptr()),
        total_m,
        gemm_n,
        gemm_k,
        reinterpret_cast<int*>(rows_for_experts.data_ptr()),
        static_cast<int*>(atomic_buffer.data_ptr()),
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
  IPEX_OP_REGISTER_DISPATCH(
      "fused_moe_gemm_persistent.moe",
      at::AtenIpexTypeXPU::fused_moe_gemm_persistent,
      c10::DispatchKey::XPU);
}

} // namespace