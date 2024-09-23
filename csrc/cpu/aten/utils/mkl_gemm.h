#pragma once
#include <ATen/ATen.h>
#include "mkl.h"

inline void _mkl_gemm(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb,
    const int& m,
    const int& n,
    const int& k,
    const float& alpha,
    const float* a,
    const int& lda,
    const float* b,
    const int& ldb,
    const float& beta,
    float* c,
    const int& ldc) {
  cblas_sgemm(
      layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void _mkl_gemm(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb,
    const int& m,
    const int& n,
    const int& k,
    const double& alpha,
    const double* a,
    const int& lda,
    const double* b,
    const int& ldb,
    const double& beta,
    double* c,
    const int& ldc) {
  cblas_dgemm(
      layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void _mkl_gemm(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb,
    const int& m,
    const int& n,
    const int& k,
    const float& alpha,
    const at::BFloat16* a,
    const int& lda,
    const at::BFloat16* b,
    const int& ldb,
    const float& beta,
    float* c,
    const int& ldc) {
  cblas_gemm_bf16bf16f32(
      layout,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      (const MKL_BF16*)(a),
      lda,
      (const MKL_BF16*)(b),
      ldb,
      beta,
      c,
      ldc);
}

inline void _mkl_gemm(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb,
    const int& m,
    const int& n,
    const int& k,
    const float& alpha,
    const at::Half* a,
    const int& lda,
    const at::Half* b,
    const int& ldb,
    const float& beta,
    float* c,
    const int& ldc) {
  TORCH_CHECK(false, "_mkl_gemm does not support FP16 yet");
}
