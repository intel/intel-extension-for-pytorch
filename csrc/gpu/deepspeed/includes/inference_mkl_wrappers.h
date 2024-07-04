/*******************************************************************************
 * Copyright 2016-2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#pragma once

#include <assert.h>
#include <stdio.h>
#include <sycl/sycl.hpp>

#include "blas_utils.h"
#include "lib_common_utils.h"

int mkl_gemm_ex(
    sycl::queue* handle,
    oneapi::mkl::transpose transa,
    oneapi::mkl::transpose transb,
    int m,
    int n,
    int k,
    const float* alpha,
    const float* beta,
    const float* A,
    const float* B,
    float* C,
    int algo,
    int b_stride = -1) try {
  const int ldb = (b_stride == -1)
      ? ((transb == oneapi::mkl::transpose::nontrans) ? k : n)
      : b_stride;
  int status = DS_CHECK_ERROR(ds::gemm(
      *handle,
      transa,
      transb,
      m,
      n,
      k,
      (const void*)alpha,
      (const void*)A,
      ds::library_data_t::real_float,
      (transa == oneapi::mkl::transpose::nontrans) ? m : k,
      (const void*)B,
      ds::library_data_t::real_float,
      ldb,
      (const void*)beta,
      C,
      ds::library_data_t::real_float,
      m,
      ds::library_data_t::real_float));

  if (status != 0) {
    fprintf(
        stderr,
        "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
        m,
        n,
        k,
        (int)status);
    return EXIT_FAILURE;
  }
  return 0;
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <typename T>
int mkl_gemm_ex(
    sycl::queue* handle,
    oneapi::mkl::transpose transa,
    oneapi::mkl::transpose transb,
    int m,
    int n,
    int k,
    const float* alpha,
    const float* beta,
    const T* A,
    const T* B,
    T* C,
    int algo,
    int b_stride = -1) try {
  const int ldb = (b_stride == -1)
      ? ((transb == oneapi::mkl::transpose::nontrans) ? k : n)
      : b_stride;
  constexpr auto mkl_dtype_16 = std::is_same<T, sycl::half>::value
      ? ds::library_data_t::real_half
      : ds::library_data_t::real_bfloat16;
  int status = DS_CHECK_ERROR(ds::gemm(
      *handle,
      transa,
      transb,
      m,
      n,
      k,
      (const void*)alpha,
      (const void*)A,
      mkl_dtype_16,
      (transa == oneapi::mkl::transpose::nontrans) ? m : k,
      (const void*)B,
      mkl_dtype_16,
      ldb,
      (const void*)beta,
      (void*)C,
      mkl_dtype_16,
      m,
      ds::library_data_t::real_float));

  if (status != 0) {
    fprintf(
        stderr,
        "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
        m,
        n,
        k,
        (int)status);
    return EXIT_FAILURE;
  }
  return 0;
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int mkl_strided_batched_gemm(
    sycl::queue* handle,
    int m,
    int n,
    int k,
    const float* alpha,
    const float* beta,
    const float* A,
    const float* B,
    float* C,
    oneapi::mkl::transpose op_A,
    oneapi::mkl::transpose op_B,
    int stride_A,
    int stride_B,
    int stride_C,
    int batch,
    int algo) try {
  int status = DS_CHECK_ERROR(ds::gemm_batch(
      *handle,
      op_A,
      op_B,
      m,
      n,
      k,
      alpha,
      A,
      ds::library_data_t::real_float,
      (op_A == oneapi::mkl::transpose::nontrans) ? m : k,
      stride_A,
      B,
      ds::library_data_t::real_float,
      (op_B == oneapi::mkl::transpose::nontrans) ? k : n,
      stride_B,
      beta,
      C,
      ds::library_data_t::real_float,
      m,
      stride_C,
      batch,
      ds::library_data_t::real_float));

  if (status != 0) {
    fprintf(
        stderr,
        "!!!! kernel execution error. (batch: %d, m: %d, n: %d, k: %d, error: %d) \n",
        batch,
        m,
        n,
        k,
        (int)status);
    return EXIT_FAILURE;
  }
  return 0;
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <typename T>
int mkl_strided_batched_gemm(
    sycl::queue* handle,
    int m,
    int n,
    int k,
    const float* alpha,
    const float* beta,
    const T* A,
    const T* B,
    T* C,
    oneapi::mkl::transpose op_A,
    oneapi::mkl::transpose op_B,
    int stride_A,
    int stride_B,
    int stride_C,
    int batch,
    int algo) try {
  constexpr auto mkl_dtype_16 = std::is_same<T, sycl::half>::value
      ? ds::library_data_t::real_half
      : ds::library_data_t::real_bfloat16;
  int status = DS_CHECK_ERROR(ds::gemm_batch(
      *handle,
      op_A,
      op_B,
      m,
      n,
      k,
      alpha,
      A,
      mkl_dtype_16,
      (op_A == oneapi::mkl::transpose::nontrans) ? m : k,
      stride_A,
      B,
      mkl_dtype_16,
      (op_B == oneapi::mkl::transpose::nontrans) ? k : n,
      stride_B,
      beta,
      C,
      mkl_dtype_16,
      m,
      stride_C,
      batch,
      ds::library_data_t::real_float));

  if (status != 0) {
    fprintf(
        stderr,
        "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
        m,
        n,
        k,
        (int)status);
    return EXIT_FAILURE;
  }

  return 0;
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
