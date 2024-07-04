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

#ifndef __BLAS_UTILS_H__
#define __BLAS_UTILS_H__

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>
#include <thread>
#include <utility>
#include <vector>
#include "lib_common_utils.h"

namespace ds {

/// Get the value of \p s.
/// Copy the data to host synchronously, then return the data.
/// \param [in] p The pointer points the data.
/// \param [in] q The queue where the memory copy should be executed.
template <typename T>
inline auto get_value(const T* s, sycl::queue& q) {
  return detail::get_value(s, q);
}

namespace detail {

template <class Ta, class Tb, class Tc, class Ts>
inline void gemm_impl(
    sycl::queue& q,
    oneapi::mkl::transpose a_trans,
    oneapi::mkl::transpose b_trans,
    int m,
    int n,
    int k,
    const void* alpha,
    const void* a,
    int lda,
    const void* b,
    int ldb,
    const void* beta,
    void* c,
    int ldc) {
#ifndef __INTEL_MKL__
  throw std::runtime_error(
      "The oneAPI Math Kernel Library (oneMKL) Interfaces "
      "Project does not support this API.");
#else
  Ts alpha_value = ds::get_value(reinterpret_cast<const Ts*>(alpha), q);
  Ts beta_value = ds::get_value(reinterpret_cast<const Ts*>(beta), q);
  auto data_a = get_memory<const Ta>(a);
  auto data_b = get_memory<const Tb>(b);
  auto data_c = get_memory<Tc>(c);
  oneapi::mkl::blas::column_major::gemm(
      q,
      a_trans,
      b_trans,
      m,
      n,
      k,
      alpha_value,
      data_a,
      lda,
      data_b,
      ldb,
      beta_value,
      data_c,
      ldc);
#endif
}

template <class Ta, class Tb, class Tc, class Ts>
inline void gemm_batch_impl(
    sycl::queue& q,
    oneapi::mkl::transpose a_trans,
    oneapi::mkl::transpose b_trans,
    int m,
    int n,
    int k,
    const void* alpha,
    const void** a,
    int lda,
    const void** b,
    int ldb,
    const void* beta,
    void** c,
    int ldc,
    int batch_size) {
  struct matrix_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts value_info[2];
    std::int64_t size_info[3];
    std::int64_t ld_info[3];
    std::int64_t groupsize_info;
  };

  Ts alpha_value = ds::get_value(reinterpret_cast<const Ts*>(alpha), q);
  Ts beta_value = ds::get_value(reinterpret_cast<const Ts*>(beta), q);

  matrix_info_t* matrix_info =
      (matrix_info_t*)std::malloc(sizeof(matrix_info_t));
  matrix_info->transpose_info[0] = a_trans;
  matrix_info->transpose_info[1] = b_trans;
  matrix_info->value_info[0] = alpha_value;
  matrix_info->value_info[1] = beta_value;
  matrix_info->size_info[0] = m;
  matrix_info->size_info[1] = n;
  matrix_info->size_info[2] = k;
  matrix_info->ld_info[0] = lda;
  matrix_info->ld_info[1] = ldb;
  matrix_info->ld_info[2] = ldc;
  matrix_info->groupsize_info = batch_size;

  sycl::event e = oneapi::mkl::blas::column_major::gemm_batch(
      q,
      matrix_info->transpose_info,
      matrix_info->transpose_info + 1,
      matrix_info->size_info,
      matrix_info->size_info + 1,
      matrix_info->size_info + 2,
      matrix_info->value_info,
      reinterpret_cast<const Ta**>(a),
      matrix_info->ld_info,
      reinterpret_cast<const Tb**>(b),
      matrix_info->ld_info + 1,
      matrix_info->value_info + 1,
      reinterpret_cast<Tc**>(c),
      matrix_info->ld_info + 2,
      1,
      &(matrix_info->groupsize_info));

  q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { std::free(matrix_info); });
  });
}

template <class Ta, class Tb, class Tc, class Ts>
inline void gemm_batch_impl(
    sycl::queue& q,
    oneapi::mkl::transpose a_trans,
    oneapi::mkl::transpose b_trans,
    int m,
    int n,
    int k,
    const void* alpha,
    const void* a,
    int lda,
    long long int stride_a,
    const void* b,
    int ldb,
    long long int stride_b,
    const void* beta,
    void* c,
    int ldc,
    long long int stride_c,
    int batch_size) {
  Ts alpha_value = ds::get_value(reinterpret_cast<const Ts*>(alpha), q);
  Ts beta_value = ds::get_value(reinterpret_cast<const Ts*>(beta), q);
  auto data_a = get_memory<const Ta>(a);
  auto data_b = get_memory<const Tb>(b);
  auto data_c = get_memory<Tc>(c);
  oneapi::mkl::blas::column_major::gemm_batch(
      q,
      a_trans,
      b_trans,
      m,
      n,
      k,
      alpha_value,
      data_a,
      lda,
      stride_a,
      data_b,
      ldb,
      stride_b,
      beta_value,
      data_c,
      ldc,
      stride_c,
      batch_size);
}

} // namespace detail

inline oneapi::mkl::transpose get_transpose(int t) {
  if (t == 0) {
    return oneapi::mkl::transpose::nontrans;
  } else if (t == 1) {
    return oneapi::mkl::transpose::trans;
  } else {
    return oneapi::mkl::transpose::conjtrans;
  }
}

/// Computes matrix-matrix product with general matrices.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] a_trans Specifies the operation applied to A.
/// \param [in] b_trans Specifies the operation applied to B.
/// \param [in] m Specifies the number of rows of the matrix op(A) and of the
/// matrix C. \param [in] n Specifies the number of columns of the matrix op(B)
/// and of the matrix C. \param [in] k Specifies the number of columns of the
/// matrix op(A) and the number of rows of the matrix op(B). \param [in] alpha
/// Scaling factor for the matrix-matrix product. \param [in] a Input matrix A.
/// \param [in] a_type Data type of the matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] b Input matrix B.
/// \param [in] b_type Data type of the matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] beta Scaling factor for matrix C.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] c_type Data type of the matrix C.
/// \param [in] ldc Leading dimension of C.
/// \param [in] scaling_type Data type of the scaling factors.
inline void gemm(
    sycl::queue& q,
    oneapi::mkl::transpose a_trans,
    oneapi::mkl::transpose b_trans,
    int m,
    int n,
    int k,
    const void* alpha,
    const void* a,
    library_data_t a_type,
    int lda,
    const void* b,
    library_data_t b_type,
    int ldb,
    const void* beta,
    void* c,
    library_data_t c_type,
    int ldc,
    library_data_t scaling_type) {
  bool matched = false;
  if (scaling_type == library_data_t::real_float &&
      c_type == library_data_t::complex_float) {
    scaling_type = library_data_t::complex_float;
  } else if (
      scaling_type == library_data_t::real_double &&
      c_type == library_data_t::complex_double) {
    scaling_type = library_data_t::complex_double;
  }

  std::uint64_t key =
      detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
  switch (key) {
    case detail::get_type_combination_id(
        library_data_t::real_float,
        library_data_t::real_float,
        library_data_t::real_float,
        library_data_t::real_float): {
      detail::gemm_impl<float, float, float, float>(
          q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_double,
        library_data_t::real_double,
        library_data_t::real_double,
        library_data_t::real_double): {
      detail::gemm_impl<double, double, double, double>(
          q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::complex_float,
        library_data_t::complex_float,
        library_data_t::complex_float,
        library_data_t::complex_float): {
      detail::gemm_impl<
          std::complex<float>,
          std::complex<float>,
          std::complex<float>,
          std::complex<float>>(
          q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::complex_double,
        library_data_t::complex_double,
        library_data_t::complex_double,
        library_data_t::complex_double): {
      detail::gemm_impl<
          std::complex<double>,
          std::complex<double>,
          std::complex<double>,
          std::complex<double>>(
          q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_half,
        library_data_t::real_half,
        library_data_t::real_half,
        library_data_t::real_half): {
      detail::gemm_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
          q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_bfloat16,
        library_data_t::real_bfloat16,
        library_data_t::real_float,
        library_data_t::real_float): {
      detail::
          gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, float, float>(
              q,
              a_trans,
              b_trans,
              m,
              n,
              k,
              alpha,
              a,
              lda,
              b,
              ldb,
              beta,
              c,
              ldc);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_half,
        library_data_t::real_half,
        library_data_t::real_float,
        library_data_t::real_float): {
      detail::gemm_impl<sycl::half, sycl::half, float, float>(
          q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_half,
        library_data_t::real_half,
        library_data_t::real_half,
        library_data_t::real_float): {
      float alpha_value =
          ds::get_value(reinterpret_cast<const float*>(alpha), q);
      float beta_value = ds::get_value(reinterpret_cast<const float*>(beta), q);
      sycl::half alpha_half(alpha_value);
      sycl::half beta_half(beta_value);
      detail::gemm_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          &alpha_half,
          a,
          lda,
          b,
          ldb,
          &beta_half,
          c,
          ldc);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_int8,
        library_data_t::real_int8,
        library_data_t::real_float,
        library_data_t::real_float): {
      detail::gemm_impl<std::int8_t, std::int8_t, float, float>(
          q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_bfloat16,
        library_data_t::real_bfloat16,
        library_data_t::real_bfloat16,
        library_data_t::real_float): {
      detail::gemm_impl<
          oneapi::mkl::bfloat16,
          oneapi::mkl::bfloat16,
          oneapi::mkl::bfloat16,
          float>(
          q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_int8,
        library_data_t::real_int8,
        library_data_t::real_int32,
        library_data_t::real_int32): {
      float alpha_float =
          ds::get_value(reinterpret_cast<const std::int32_t*>(alpha), q);
      float beta_float =
          ds::get_value(reinterpret_cast<const std::int32_t*>(beta), q);
      detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, float>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          &alpha_float,
          a,
          lda,
          b,
          ldb,
          &beta_float,
          c,
          ldc);
      break;
    }
    default:
      throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Computes a batch of matrix-matrix product with general matrices.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] a_trans Specifies the operation applied to A.
/// \param [in] b_trans Specifies the operation applied to B.
/// \param [in] m Specifies the number of rows of the matrix op(A) and of the
/// matrix C. \param [in] n Specifies the number of columns of the matrix op(B)
/// and of the matrix C. \param [in] k Specifies the number of columns of the
/// matrix op(A) and the number of rows of the matrix op(B). \param [in] alpha
/// Scaling factor for the matrix-matrix product. \param [in] a Input matrix A.
/// \param [in] a_type Data type of the matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] b Input matrix B.
/// \param [in] b_type Data type of the matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] beta Scaling factor for matrix C.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] c_type Data type of the matrix C.
/// \param [in] ldc Leading dimension of C.
/// \param [in] batch_size Specifies the number of matrix multiply operations to
/// perform. \param [in] scaling_type Data type of the scaling factors.
inline void gemm_batch(
    sycl::queue& q,
    oneapi::mkl::transpose a_trans,
    oneapi::mkl::transpose b_trans,
    int m,
    int n,
    int k,
    const void* alpha,
    const void* a[],
    library_data_t a_type,
    int lda,
    const void* b[],
    library_data_t b_type,
    int ldb,
    const void* beta,
    void* c[],
    library_data_t c_type,
    int ldc,
    int batch_size,
    library_data_t scaling_type) {
  bool matched = false;
  if (scaling_type == library_data_t::real_float &&
      c_type == library_data_t::complex_float) {
    scaling_type = library_data_t::complex_float;
  } else if (
      scaling_type == library_data_t::real_double &&
      c_type == library_data_t::complex_double) {
    scaling_type = library_data_t::complex_double;
  }

  std::uint64_t key =
      detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
  switch (key) {
    case detail::get_type_combination_id(
        library_data_t::real_float,
        library_data_t::real_float,
        library_data_t::real_float,
        library_data_t::real_float): {
      detail::gemm_batch_impl<float, float, float, float>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          b,
          ldb,
          beta,
          c,
          ldc,
          batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_double,
        library_data_t::real_double,
        library_data_t::real_double,
        library_data_t::real_double): {
      detail::gemm_batch_impl<double, double, double, double>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          b,
          ldb,
          beta,
          c,
          ldc,
          batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::complex_float,
        library_data_t::complex_float,
        library_data_t::complex_float,
        library_data_t::complex_float): {
      detail::gemm_batch_impl<
          std::complex<float>,
          std::complex<float>,
          std::complex<float>,
          std::complex<float>>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          b,
          ldb,
          beta,
          c,
          ldc,
          batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::complex_double,
        library_data_t::complex_double,
        library_data_t::complex_double,
        library_data_t::complex_double): {
      detail::gemm_batch_impl<
          std::complex<double>,
          std::complex<double>,
          std::complex<double>,
          std::complex<double>>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          b,
          ldb,
          beta,
          c,
          ldc,
          batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_half,
        library_data_t::real_half,
        library_data_t::real_half,
        library_data_t::real_half): {
      detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          b,
          ldb,
          beta,
          c,
          ldc,
          batch_size);
      break;
    }
#ifdef __INTEL_MKL__
    case detail::get_type_combination_id(
        library_data_t::real_bfloat16,
        library_data_t::real_bfloat16,
        library_data_t::real_bfloat16,
        library_data_t::real_float): {
      detail::gemm_batch_impl<
          oneapi::mkl::bfloat16,
          oneapi::mkl::bfloat16,
          oneapi::mkl::bfloat16,
          float>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          b,
          ldb,
          beta,
          c,
          ldc,
          batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_bfloat16,
        library_data_t::real_bfloat16,
        library_data_t::real_float,
        library_data_t::real_float): {
      detail::gemm_batch_impl<
          oneapi::mkl::bfloat16,
          oneapi::mkl::bfloat16,
          float,
          float>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          b,
          ldb,
          beta,
          c,
          ldc,
          batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_int8,
        library_data_t::real_int8,
        library_data_t::real_int32,
        library_data_t::real_int32): {
      float alpha_float =
          ds::get_value(reinterpret_cast<const std::int32_t*>(alpha), q);
      float beta_float =
          ds::get_value(reinterpret_cast<const std::int32_t*>(beta), q);
      detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, float>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          &alpha_float,
          a,
          lda,
          b,
          ldb,
          &beta_float,
          c,
          ldc,
          batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_int8,
        library_data_t::real_int8,
        library_data_t::real_float,
        library_data_t::real_float): {
      detail::gemm_batch_impl<std::int8_t, std::int8_t, float, float>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          b,
          ldb,
          beta,
          c,
          ldc,
          batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_half,
        library_data_t::real_half,
        library_data_t::real_float,
        library_data_t::real_float): {
      detail::gemm_batch_impl<sycl::half, sycl::half, float, float>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          b,
          ldb,
          beta,
          c,
          ldc,
          batch_size);
      break;
    }
#endif
    case detail::get_type_combination_id(
        library_data_t::real_half,
        library_data_t::real_half,
        library_data_t::real_half,
        library_data_t::real_float): {
      float alpha_value =
          ds::get_value(reinterpret_cast<const float*>(alpha), q);
      float beta_value = ds::get_value(reinterpret_cast<const float*>(beta), q);
      sycl::half alpha_half(alpha_value);
      sycl::half beta_half(beta_value);
      detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          &alpha_half,
          a,
          lda,
          b,
          ldb,
          &beta_half,
          c,
          ldc,
          batch_size);
      break;
    }
    default:
      throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Computes a batch of matrix-matrix product with general matrices.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] a_trans Specifies the operation applied to A.
/// \param [in] b_trans Specifies the operation applied to B.
/// \param [in] m Specifies the number of rows of the matrix op(A) and of the
/// matrix C. \param [in] n Specifies the number of columns of the matrix op(B)
/// and of the matrix C. \param [in] k Specifies the number of columns of the
/// matrix op(A) and the number of rows of the matrix op(B). \param [in] alpha
/// Scaling factor for the matrix-matrix product. \param [in] a Input matrix A.
/// \param [in] a_type Data type of the matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] stride_a Stride between the different A matrices.
/// \param [in] b Input matrix B.
/// \param [in] b_type Data type of the matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] stride_b Stride between the different B matrices.
/// \param [in] beta Scaling factor for matrix C.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] c_type Data type of the matrix C.
/// \param [in] ldc Leading dimension of C.
/// \param [in] stride_c Stride between the different C matrices.
/// \param [in] batch_size Specifies the number of matrix multiply operations to
/// perform. \param [in] scaling_type Data type of the scaling factors.
inline void gemm_batch(
    sycl::queue& q,
    oneapi::mkl::transpose a_trans,
    oneapi::mkl::transpose b_trans,
    int m,
    int n,
    int k,
    const void* alpha,
    const void* a,
    library_data_t a_type,
    int lda,
    long long int stride_a,
    const void* b,
    library_data_t b_type,
    int ldb,
    long long int stride_b,
    const void* beta,
    void* c,
    library_data_t c_type,
    int ldc,
    long long int stride_c,
    int batch_size,
    library_data_t scaling_type) {
  bool matched = false;
  if (scaling_type == library_data_t::real_float &&
      c_type == library_data_t::complex_float) {
    scaling_type = library_data_t::complex_float;
  } else if (
      scaling_type == library_data_t::real_double &&
      c_type == library_data_t::complex_double) {
    scaling_type = library_data_t::complex_double;
  }

  std::uint64_t key =
      detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
  switch (key) {
    case detail::get_type_combination_id(
        library_data_t::real_float,
        library_data_t::real_float,
        library_data_t::real_float,
        library_data_t::real_float): {
      detail::gemm_batch_impl<float, float, float, float>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          stride_a,
          b,
          ldb,
          stride_b,
          beta,
          c,
          ldc,
          stride_c,
          batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_double,
        library_data_t::real_double,
        library_data_t::real_double,
        library_data_t::real_double): {
      detail::gemm_batch_impl<double, double, double, double>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          stride_a,
          b,
          ldb,
          stride_b,
          beta,
          c,
          ldc,
          stride_c,
          batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::complex_float,
        library_data_t::complex_float,
        library_data_t::complex_float,
        library_data_t::complex_float): {
      detail::gemm_batch_impl<
          std::complex<float>,
          std::complex<float>,
          std::complex<float>,
          std::complex<float>>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          stride_a,
          b,
          ldb,
          stride_b,
          beta,
          c,
          ldc,
          stride_c,
          batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::complex_double,
        library_data_t::complex_double,
        library_data_t::complex_double,
        library_data_t::complex_double): {
      detail::gemm_batch_impl<
          std::complex<double>,
          std::complex<double>,
          std::complex<double>,
          std::complex<double>>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          stride_a,
          b,
          ldb,
          stride_b,
          beta,
          c,
          ldc,
          stride_c,
          batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_half,
        library_data_t::real_half,
        library_data_t::real_half,
        library_data_t::real_half): {
      detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          stride_a,
          b,
          ldb,
          stride_b,
          beta,
          c,
          ldc,
          stride_c,
          batch_size);
      break;
    }
#ifdef __INTEL_MKL__
    case detail::get_type_combination_id(
        library_data_t::real_bfloat16,
        library_data_t::real_bfloat16,
        library_data_t::real_bfloat16,
        library_data_t::real_float): {
      detail::gemm_batch_impl<
          oneapi::mkl::bfloat16,
          oneapi::mkl::bfloat16,
          oneapi::mkl::bfloat16,
          float>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          stride_a,
          b,
          ldb,
          stride_b,
          beta,
          c,
          ldc,
          stride_c,
          batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_bfloat16,
        library_data_t::real_bfloat16,
        library_data_t::real_float,
        library_data_t::real_float): {
      detail::gemm_batch_impl<
          oneapi::mkl::bfloat16,
          oneapi::mkl::bfloat16,
          float,
          float>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          stride_a,
          b,
          ldb,
          stride_b,
          beta,
          c,
          ldc,
          stride_c,
          batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_int8,
        library_data_t::real_int8,
        library_data_t::real_int32,
        library_data_t::real_int32): {
      detail::
          gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, std::int32_t>(
              q,
              a_trans,
              b_trans,
              m,
              n,
              k,
              alpha,
              a,
              lda,
              stride_a,
              b,
              ldb,
              stride_b,
              beta,
              c,
              ldc,
              stride_c,
              batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_int8,
        library_data_t::real_int8,
        library_data_t::real_float,
        library_data_t::real_float): {
      detail::gemm_batch_impl<std::int8_t, std::int8_t, float, float>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          stride_a,
          b,
          ldb,
          stride_b,
          beta,
          c,
          ldc,
          stride_c,
          batch_size);
      break;
    }
    case detail::get_type_combination_id(
        library_data_t::real_half,
        library_data_t::real_half,
        library_data_t::real_float,
        library_data_t::real_float): {
      detail::gemm_batch_impl<sycl::half, sycl::half, float, float>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          alpha,
          a,
          lda,
          stride_a,
          b,
          ldb,
          stride_b,
          beta,
          c,
          ldc,
          stride_c,
          batch_size);
      break;
    }
#endif
    case detail::get_type_combination_id(
        library_data_t::real_half,
        library_data_t::real_half,
        library_data_t::real_half,
        library_data_t::real_float): {
      float alpha_value =
          ds::get_value(reinterpret_cast<const float*>(alpha), q);
      float beta_value = ds::get_value(reinterpret_cast<const float*>(beta), q);
      sycl::half alpha_half(alpha_value);
      sycl::half beta_half(beta_value);
      detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
          q,
          a_trans,
          b_trans,
          m,
          n,
          k,
          &alpha_half,
          a,
          lda,
          stride_a,
          b,
          ldb,
          stride_b,
          &beta_half,
          c,
          ldc,
          stride_c,
          batch_size);
      break;
    }
    default:
      throw std::runtime_error("the combination of data type is unsupported");
  }
}

} // namespace ds
#endif // __BLAS_UTILS_H__
