#pragma once
#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>
#include <libxsmm_rng.h>

template<typename T> struct xsmmType { };
template<> struct xsmmType<float> {
  using type = libxsmm_smmfunction;
  using dtype = float;
};
template<> struct xsmmType<at::BFloat16> {
  using type = libxsmm_bmmfunction;
  using dtype = libxsmm_bfloat16;
};

template<typename T>
using xsmm_type = typename xsmmType<T>::type;
template<typename T>
using xsmm_dtype = typename xsmmType<T>::dtype;

template<typename T>
inline xsmm_type<T> get_mm_kernel(int32_t M, int32_t N, int32_t K) { }

template<>
inline libxsmm_smmfunction get_mm_kernel<float>(int32_t M, int32_t N, int32_t K) {
  float alpha = 1.0;
  float beta = 0.0;
  auto flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  auto mm_kernel = libxsmm_smmdispatch(N, M, K, NULL, NULL, NULL, &alpha, &beta, &flags, NULL);
  return mm_kernel;
}

template<>
inline libxsmm_bmmfunction get_mm_kernel<at::BFloat16>(int32_t M, int32_t N, int32_t K) {
  float alpha = 1.0;
  float beta = 0.0;
  auto flags = LIBXSMM_GEMM_FLAGS('N', 'N') | LIBXSMM_GEMM_FLAG_VNNI_A;
  auto mm_kernel = libxsmm_bmmdispatch(N, M, K, NULL, NULL, NULL, &alpha, &beta, &flags, NULL);
  return mm_kernel;
}

inline libxsmm_xtransfunction get_tr_kernel(int M, int N, int LDO) {
  libxsmm_xtransfunction tr_kernel;
  libxsmm_descriptor_blob blob;
  libxsmm_trans_descriptor *tr_desc;
  tr_desc = libxsmm_trans_descriptor_init(&blob, sizeof(float), M, N, LDO);
  tr_kernel = libxsmm_dispatch_trans(tr_desc);
  return tr_kernel;
}
